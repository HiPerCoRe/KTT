#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <iterator>
#include <optional>
#include <sqlite3.h>
#include <string>
#include <utility>

#include <Api/Info/DatabaseTuningInfo.h>
#include <Api/KttException.h>
#include <Database/Local/Database.h>
#include <Database/Local/Repository/Device/DeviceRepository.h>
#include <Database/Local/Repository/Result/ResultRepository.h>
#include <Database/Local/Repository/Run/RunRepository.h>
#include <Database/Local/Repository/Source/SourceRepository.h>
#include <Database/Local/Repository/Space/SpaceRepository.h>
#include <Database/Local/Repository/Tuner/TunerRepository.h>
#include <Database/Local/Schema/Schema.h>
#include <Database/Local/Utility/TransactionGuard.h>
#include <Database/Utility/DataDirectory.h>
#include <Database/Utility/TimeUnitGuard.h>
#include <Database/Utility/TuningInfoValidation.h>
#include <Output/OutputFormat.h>
#include <Output/TimeConfiguration/TimeConfiguration.h>
#include <Utility/Logger/Logger.h>

namespace ktt::db
{

// SQLite expects file names in UTF-8. path::string() uses the native narrow encoding instead, which on Windows is the
// ANSI code page and breaks (or throws) on non-ASCII paths such as user names with diacritics.
static std::string ToUtf8(const std::filesystem::path &path)
{
    return path.u8string();
}

// File behind a connection's main database; empty for in-memory and temporary databases
static std::filesystem::path GetConnectionFilePath(sqlite3 *connection)
{
    const char *filename = sqlite3_db_filename(connection, "main");
    return (filename != nullptr && *filename != '\0') ? std::filesystem::u8path(filename) : std::filesystem::path();
}

// True when both connections operate on the same database: the same handle, or the same file (also when reached
// through a relative path, symlink or hard link)
static bool IsSameDatabase(sqlite3 *first, sqlite3 *second)
{
    if (first == second)
        return true;

    const std::filesystem::path firstPath = GetConnectionFilePath(first);
    const std::filesystem::path secondPath = GetConnectionFilePath(second);

    if (firstPath.empty() || secondPath.empty())
        return false;

    std::error_code error;
    return std::filesystem::equivalent(firstPath, secondPath, error);
}

Database::Database() : m_Connection(nullptr)
{
    try
    {
        m_DatabasePath = GetDefaultDataDirectory();
    }
    catch (const KttException &exception)
    {
        throw KttException(std::string(exception.what()) + "; construct the Database with an explicit path");
    }

    std::filesystem::create_directories(m_DatabasePath);
    m_DatabasePath /= "ktt.db";

    OpenOrCreateDatabase();
}

Database::Database(std::filesystem::path databasePath) : m_DatabasePath(std::move(databasePath)), m_Connection(nullptr)
{
    OpenOrCreateDatabase();
}

Database::Database(sqlite3 *connection) : m_Connection(connection), m_OwnsConnection(false)
{
    if (m_Connection == nullptr)
        throw KttException("Cannot construct Database from a null SQLite connection");

    const std::filesystem::path filePath = GetConnectionFilePath(m_Connection);
    m_DatabasePath = filePath.empty() ? std::filesystem::path(":memory:") : filePath;

    ktt::Logger::LogInfo("Initializing database from existing SQLite connection at " + ToUtf8(m_DatabasePath));
    sqlite3_exec(m_Connection, "PRAGMA foreign_keys = ON;", nullptr, nullptr, nullptr);
    Schema::CreateIfNotExists(m_Connection);
}


Database::~Database()
{
    CloseDatabase();
}

void Database::OpenOrCreateDatabase() const
{
    if (m_Connection != nullptr)
        return;

    if (m_DatabasePath == ":memory:")
        ktt::Logger::LogInfo("Creating in-memory database");
    else if (std::filesystem::exists(m_DatabasePath))
        ktt::Logger::LogInfo("Opening database at " + ToUtf8(m_DatabasePath));
    else
        ktt::Logger::LogInfo("Creating new database at " + ToUtf8(m_DatabasePath));

    const int db = sqlite3_open(ToUtf8(m_DatabasePath).c_str(), &m_Connection);

    if (db != SQLITE_OK)
    {
        std::string error = sqlite3_errmsg(m_Connection);
        sqlite3_close(m_Connection);
        m_Connection = nullptr;
        throw std::runtime_error("Failed to open or create database: " + error);
    }
    sqlite3_exec(m_Connection, "PRAGMA foreign_keys = ON;", nullptr, nullptr, nullptr);
    Schema::CreateIfNotExists(m_Connection);
}

void Database::CloseDatabase() const
{
    if (m_OwnsConnection && m_Connection != nullptr)
    {
        sqlite3_close(m_Connection);
        m_Connection = nullptr;
    }
}

void Database::SaveResults(const TuningInfo &tuningInfo, std::vector<KernelResult> results, SaveOptions option) const
{
    ValidateTuningInfo(tuningInfo);

    try
    {
        // The run records the current time unit and its results are serialized in it. The lock keeps other database
        // threads from switching the unit meanwhile. A reader can hold an SQLite lock while it waits for this one, but
        // no busy timeout is set, so SQLite answers this save with SQLITE_BUSY rather than waiting: no deadlock.
        const auto lock = LockTimeUnit();
        TransactionGuard transaction(m_Connection);

        const auto source = SourceRepository::GetOrCreateSource(
            m_Connection,
            {std::nullopt, tuningInfo.spaceInfo.sourceFingerprint, tuningInfo.spaceInfo.sourceName}
        );

        const auto space = SpaceRepository::GetOrCreateSpace(
            m_Connection,
            {std::nullopt, // space Id
             *source.id,
             tuningInfo.spaceInfo.parameterFingerprint,
             tuningInfo.spaceInfo.spaceFingerprint,
             tuningInfo.spaceInfo.spaceName}
        );

        const auto device = DeviceRepository::GetOrCreateDevice(
            m_Connection,
            Device::FromDeviceInfo(tuningInfo.device)
        );

        const auto tuner = TunerRepository::GetOrCreateTuner(
            m_Connection,
            {std::nullopt, tuningInfo.tuner.name, tuningInfo.tuner.version}
        );

        const size_t runId = RunRepository::CreateRun(
            m_Connection,
            {std::nullopt, // run Id
             *space.id,
             *device.id,
             *tuner.id,
             option.format,
             tuningInfo.inputData,
             // The results below are serialized with the time unit active now, so the run records that unit.
             ktt::TimeConfiguration::GetInstance().GetTimeUnit()}
        );

        ResultRepository::CreateResults(m_Connection, runId, results, option.format, option.indent);
        transaction.Commit();
    } catch (...)
    {
        throw;
    }
}

std::vector<KernelResult> Database::SimpleGetBestResults(const TuningInfo &t, uint32_t limit) const
{
    ValidateTuningInfo(t);

    const auto source = SourceRepository::GetSource(m_Connection, t.spaceInfo.sourceFingerprint);
    if (source == std::nullopt)
    {
        ktt::Logger::LogInfo("No results found for this source");
        return {};
    }

    const auto space = SpaceRepository::GetSpace(
        m_Connection,
        {std::nullopt, // space Id
         source.value().id.value(),
         t.spaceInfo.parameterFingerprint,
         t.spaceInfo.spaceFingerprint,
         std::nullopt} // name, not used by the lookup
    );
    if (space == std::nullopt)
    {
        ktt::Logger::LogInfo("No results found for this source and parameter combination");
        return {};
    }

    return ResultRepository::SimpleGetBestResults(
        m_Connection,
        space.value().id.value(),
        Device::FromDeviceInfo(t.device),
        limit
    );
}

std::vector<KernelResult> Database::GetBestResults(const GetBestResultsQuery &query) const
{
    if (query.limit <= 0)
        return {};

    const auto source = SourceRepository::GetSource(m_Connection, query.source.sourceFingerprint);
    if (source == std::nullopt)
    {
        ktt::Logger::LogInfo("No results found for this source");
        return {};
    }

    const auto space = SpaceRepository::GetSpace(
        m_Connection,
        {std::nullopt, // space Id
         source.value().id.value(),
         query.source.parameterFingerprint,
         query.source.spaceFingerprint,
         std::nullopt} // name, not used by the lookup
    );
    if (space == std::nullopt)
    {
        ktt::Logger::LogInfo("No results found for this source and parameter combination");
        return {};
    }

    // Each result keeps its stored kernel duration (tuning_result.duration), the key the SQL orders by.
    std::vector<StoredResult> bestResults;
    size_t offset = 0;

    for (auto runs = RunRepository::GetRunsBySpaceId(m_Connection, space.value().id.value(), offset, RunBatchSize);
         !runs.empty();
         runs = RunRepository::GetRunsBySpaceId(m_Connection, space.value().id.value(), offset, RunBatchSize))
    {

        offset += runs.size();

        std::vector<size_t> runIds;
        runIds.reserve(runs.size());

        for (const auto &run : runs)
        {
            bool keep = true;
            if (query.devicePredicate)
                keep = keep && (*query.devicePredicate)(run.deviceInfo);
            if (query.inputPredicate)
            {
                if (run.inputData)
                    keep = keep && (*query.inputPredicate)(*run.inputData);
                else
                    keep = false;
            }

            if (keep)
                runIds.push_back(run.runId);
        }

        if (runIds.empty())
            continue;

        auto batchResults = ResultRepository::ResultsByRunIds(m_Connection, runIds, query.limit);

        if (batchResults.empty())
            continue;

        // Both ranges are sorted by the stored kernel duration: bestResults by earlier merges, batchResults by the SQL.
        // The merge uses that same key and is stable, so equal durations keep their order and earlier batches win ties.
        const auto oldSize = static_cast<std::ptrdiff_t>(bestResults.size());
        bestResults.insert(bestResults.end(), std::make_move_iterator(batchResults.begin()),
            std::make_move_iterator(batchResults.end()));
        std::inplace_merge(bestResults.begin(), bestResults.begin() + oldSize, bestResults.end(),
            [](const StoredResult &left, const StoredResult &right) { return left.duration < right.duration; });

        if (bestResults.size() > query.limit)
            bestResults.erase(bestResults.begin() + query.limit, bestResults.end());
    }

    std::vector<KernelResult> results;
    results.reserve(bestResults.size());
    for (auto &best : bestResults)
        results.push_back(std::move(best.result));

    return results;
}

std::optional<SourceStats> Database::GetStatsForSource(const size_t sourceFingerprint) const
{
    return SourceRepository::GetStatsForSource(m_Connection, sourceFingerprint);
}

size_t Database::SyncFrom(const Database &other) const
{
    sqlite3 *source = other.m_Connection;
    const std::string otherPath = ToUtf8(other.m_DatabasePath);

    if (IsSameDatabase(source, m_Connection))
        throw KttException("Cannot sync: database cannot be synced from itself: " + otherPath);

    ktt::Logger::LogInfo("Syncing runs from " + otherPath + " into " + ToUtf8(m_DatabasePath));

    size_t inserted = 0;
    size_t total = 0;
    TransactionGuard transaction(m_Connection);

    for (auto records = RunRepository::GetRuns(source, total, RunBatchSize); !records.empty();
         records = RunRepository::GetRuns(source, total, RunBatchSize))
    {
        total += records.size();

        for (const auto &record : records)
        {
            if (RunRepository::RunExists(m_Connection, record.guid))
                continue;

            const auto sourceRow = SourceRepository::GetOrCreateSource(
                m_Connection,
                {std::nullopt, record.sourceFingerprint, record.sourceName}
            );

            const auto space = SpaceRepository::GetOrCreateSpace(
                m_Connection,
                {std::nullopt, *sourceRow.id, record.parameterFingerprint, record.spaceFingerprint, record.spaceName}
            );

            const auto device = DeviceRepository::GetOrCreateDevice(
                m_Connection,
                Device::FromDeviceInfo(record.deviceInfo)
            );

            const auto tuner = TunerRepository::GetOrCreateTuner(
                m_Connection,
                {std::nullopt, record.tuner.name, record.tuner.version}
            );

            const size_t newRunId = RunRepository::CreateRunWithGuid(
                m_Connection,
                {std::nullopt, // run Id
                 *space.id,
                 *device.id,
                 *tuner.id,
                 record.outputFormat,
                 record.inputData,
                 record.timeUnit},
                record.guid,
                record.createdAt
            );

            const auto rawResults = ResultRepository::GetRawResultsByRunId(source, record.runId);
            ResultRepository::InsertRawResults(m_Connection, newRunId, rawResults);

            ++inserted;
        }
    }

    transaction.Commit();

    ktt::Logger::LogInfo(
        "Database sync: " + std::to_string(inserted) + " new run(s) copied, " + std::to_string(total - inserted) +
        " already present"
    );

    return inserted;
}

} // namespace ktt::db
