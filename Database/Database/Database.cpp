#include <cstdlib>
#include <filesystem>
#include <optional>
#include <sqlite3.h>
#include <string>
#include <utility>

#include <Api/Info/DatabaseTuningInfo.h>
#include <Api/KttException.h>
#include <Database/Database.h>
#include <Database/Repository/Device/DeviceRepository.h>
#include <Database/Repository/Result/ResultRepository.h>
#include <Database/Repository/Run/RunRepository.h>
#include <Database/Repository/Source/SourceRepository.h>
#include <Database/Repository/Space/SpaceRepository.h>
#include <Database/Repository/Tuner/TunerRepository.h>
#include <Database/Schema/Schema.h>
#include <Database/Utility/TransactionGuard.h>
#include <Database/Utility/TuningInfoValidation.h>
#include <Output/OutputFormat.h>
#include <Utility/Logger/Logger.h>

namespace ktt::db
{

// Directory of the default database: %LOCALAPPDATA%\ktt on Windows, $XDG_DATA_HOME/ktt or ~/.local/share/ktt elsewhere.
static std::filesystem::path GetDefaultDatabaseDirectory()
{
#if defined(_MSC_VER)
    // The wide variant is used because the narrow getenv returns the ANSI code page, which cannot represent every user
    // name. C4996 (deprecated) is irrelevant here, the value is copied into the path right away.
    #pragma warning(suppress : 4996)
    if (const wchar_t* localAppData = _wgetenv(L"LOCALAPPDATA"); localAppData != nullptr && *localAppData != L'\0')
        return std::filesystem::path(localAppData) / "ktt";
#else
    if (const char* dataHome = std::getenv("XDG_DATA_HOME"); dataHome != nullptr && *dataHome != '\0')
        return std::filesystem::path(dataHome) / "ktt";

    if (const char* home = std::getenv("HOME"); home != nullptr && *home != '\0')
        return std::filesystem::path(home) / ".local" / "share" / "ktt";
#endif // _MSC_VER

    throw KttException("Cannot determine the default database location, construct the Database with an explicit path");
}

// SQLite expects file names in UTF-8. path::string() uses the native narrow encoding instead, which on Windows is the
// ANSI code page and breaks (or throws) on non-ASCII paths such as user names with diacritics.
static std::string ToUtf8(const std::filesystem::path& path)
{
    return path.u8string();
}

Database::Database() : m_Connection(nullptr)
{
    m_DatabasePath = GetDefaultDatabaseDirectory();
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

    const char *filename = sqlite3_db_filename(m_Connection, "main");
    m_DatabasePath = (filename != nullptr) ? std::filesystem::u8path(filename) : std::filesystem::path();

    ktt::Logger::LogInfo("Initializing database from existing SQLite connection");
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
        TransactionGuard transaction(m_Connection);

        const auto source = SourceRepository::GetOrCreateSource(
            m_Connection,
            {std::nullopt, tuningInfo.spaceInfo.sourceFingerprint}
        );

        const auto space = SpaceRepository::GetOrCreateSpace(
            m_Connection,
            {std::nullopt, // space Id
             *source.id,
             tuningInfo.spaceInfo.parameterFingerprint,
             tuningInfo.spaceInfo.spaceFingerprint}
        );

        const auto device = DeviceRepository::GetOrCreateDevice(m_Connection, Device::FromDeviceInfo(tuningInfo.device));

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
             tuningInfo.inputData}
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
         t.spaceInfo.spaceFingerprint}
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
         query.source.spaceFingerprint}
    );
    if (space == std::nullopt)
    {
        ktt::Logger::LogInfo("No results found for this source and parameter combination");
        return {};
    }

    std::vector<KernelResult> bestResults;
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

        bestResults.insert(bestResults.end(), batchResults.begin(), batchResults.end());

        std::sort(bestResults.begin(), bestResults.end(), [](const KernelResult &left, const KernelResult &right) {
            return left.GetTotalDuration() < right.GetTotalDuration();
        });

        if (bestResults.size() > query.limit)
            bestResults.resize(query.limit);
    }

    return bestResults;
}

std::optional<SourceStats> Database::GetStatsForSource(const size_t sourceFingerprint) const
{
    return SourceRepository::GetStatsForSource(m_Connection, sourceFingerprint);
}

size_t Database::SyncFromFile(const std::filesystem::path &sourceDatabase) const
{
    if (!std::filesystem::exists(sourceDatabase))
        throw KttException("Cannot sync: database file does not exist: " + ToUtf8(sourceDatabase));

    sqlite3 *source = nullptr;
    const int open = sqlite3_open_v2(ToUtf8(sourceDatabase).c_str(), &source, SQLITE_OPEN_READONLY, nullptr);

    if (open != SQLITE_OK)
    {
        const std::string error = sqlite3_errmsg(source);
        sqlite3_close(source);
        throw KttException("Failed to open source database for sync: " + error);
    }

    ktt::Logger::LogInfo("Syncing runs from " + ToUtf8(sourceDatabase) + " into " + ToUtf8(m_DatabasePath));

    size_t inserted = 0;
    try
    {
        TransactionGuard transaction(m_Connection);

        const auto records = RunRepository::GetAllRuns(source);

        for (const auto &record : records)
        {
            if (RunRepository::RunExists(m_Connection, record.guid))
                continue;

            const auto sourceRow = SourceRepository::GetOrCreateSource(
                m_Connection,
                {std::nullopt, record.sourceFingerprint}
            );

            const auto space = SpaceRepository::GetOrCreateSpace(
                m_Connection,
                {std::nullopt, *sourceRow.id, record.parameterFingerprint, record.spaceFingerprint}
            );

            const auto device = DeviceRepository::GetOrCreateDevice(m_Connection, Device::FromDeviceInfo(record.deviceInfo));

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
                 record.inputData},
                record.guid,
                record.createdAt
            );

            const auto rawResults = ResultRepository::GetRawResultsByRunId(source, record.runId);
            ResultRepository::InsertRawResults(m_Connection, newRunId, rawResults);

            ++inserted;
        }

        transaction.Commit();

        ktt::Logger::LogInfo(
            "Database sync: " + std::to_string(inserted) + " new run(s) copied, " +
            std::to_string(records.size() - inserted) + " already present"
        );
    } catch (...)
    {
        sqlite3_close(source);
        throw;
    }

    sqlite3_close(source);
    return inserted;
}

} // namespace ktt::db
