#include <string>

#include <Database/Repository/Device/DeviceRepository.h>
#include <Database/Repository/Run/RunRepository.h>
#include <Database/Utility/Statement.h>
#include <Database/Utility/Uuid.h>

namespace ktt::db
{

// Device and tuner columns shared by the run SELECT statements, in the order read by ReadDeviceAndTunerColumns.
static const std::string DeviceAndTunerColumns =
    "device_info.name, device_info.vendor, device_info.type, "
    "device_api.compute_api_id, device_api.version_major, device_api.version_minor, device_api.extensions, "
    "device.driver_version, device.device_identifier, tuner.name, tuner.version";

// Joins from tuning_run to the tables of DeviceAndTunerColumns.
static const std::string DeviceAndTunerJoins =
    " JOIN device ON device.id = tuning_run.device_id"
    " JOIN device_info ON device_info.id = device.device_info_id"
    " JOIN device_api ON device_api.id = device.device_api_id"
    " JOIN tuner ON tuner.id = tuning_run.tuner_id ";

// Reads the DeviceAndTunerColumns starting at column index first.
static void ReadDeviceAndTunerColumns(const Statement& statement, const int first, DeviceInfo& deviceInfo, TunerInfo& tuner)
{
    deviceInfo.name = statement.GetText(first);
    deviceInfo.vendor = statement.GetText(first + 1);
    deviceInfo.type = statement.GetText(first + 2);
    deviceInfo.computeApi = static_cast<ComputeApi>(statement.GetInt(first + 3));

    // Placeholders stored instead of NULL (see DeviceRepository.h) are read back as unset values.
    if (const int major = statement.GetInt(first + 4); major != NoCudaComputeCapability)
        deviceInfo.cudaComputeCapabilityMajor = static_cast<uint32_t>(major);
    if (const int minor = statement.GetInt(first + 5); minor != NoCudaComputeCapability)
        deviceInfo.cudaComputeCapabilityMinor = static_cast<uint32_t>(minor);
    if (auto extensions = statement.GetText(first + 6); !extensions.empty())
        deviceInfo.extensions = std::move(extensions);

    deviceInfo.driverVersion = statement.GetText(first + 7);

    if (auto identifier = statement.GetText(first + 8); !identifier.empty())
        deviceInfo.deviceIdentifier = std::move(identifier);

    tuner.name = statement.GetText(first + 9);
    tuner.version = statement.GetText(first + 10);
}

size_t RunRepository::CreateRun(sqlite3* connection, const Run& run)
{
    Statement statement(connection, R"(
        INSERT INTO tuning_run (guid, space_id, device_id, tuner_id, output_format_id, input_data)
        VALUES (?, ?, ?, ?, ?, ?)
    )");

    statement.BindUuid(1, UuidGenerator::GenerateUuid());
    statement.BindInt64(2, static_cast<int64_t>(run.spaceId));
    statement.BindInt64(3, static_cast<int64_t>(run.deviceId));
    statement.BindInt64(4, static_cast<int64_t>(run.tunerId));
    statement.BindInt(5, static_cast<int>(run.outputFormat));
    statement.BindOptionalText(6, run.inputData);
    statement.Execute();

    return sqlite3_last_insert_rowid(connection);
}

size_t RunRepository::CreateRunWithGuid(
    sqlite3* connection, const Run& run, const uuid& guid, const std::string& createdAt
)
{
    Statement statement(connection, R"(
        INSERT INTO tuning_run (guid, space_id, device_id, tuner_id, output_format_id, input_data, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    )");

    statement.BindUuid(1, guid);
    statement.BindInt64(2, static_cast<int64_t>(run.spaceId));
    statement.BindInt64(3, static_cast<int64_t>(run.deviceId));
    statement.BindInt64(4, static_cast<int64_t>(run.tunerId));
    statement.BindInt(5, static_cast<int>(run.outputFormat));
    statement.BindOptionalText(6, run.inputData);
    statement.BindText(7, createdAt);
    statement.Execute();

    return sqlite3_last_insert_rowid(connection);
}

bool RunRepository::RunExists(sqlite3* connection, const uuid& guid)
{
    Statement statement(connection, R"(
        SELECT 1 FROM tuning_run WHERE guid = ? LIMIT 1
    )");

    statement.BindUuid(1, guid);
    return statement.Step();
}

std::vector<RunSyncRecord> RunRepository::GetAllRuns(sqlite3* connection)
{
    Statement statement(connection,
        "SELECT tuning_run.id, tuning_run.guid, tuning_source.source_fingerprint, "
        "tuning_space.parameter_fingerprint, tuning_space.space_fingerprint, "
        "tuning_run.output_format_id, tuning_run.input_data, tuning_run.created_at, " +
        DeviceAndTunerColumns +
        " FROM tuning_run"
        " JOIN tuning_space ON tuning_space.id = tuning_run.space_id"
        " JOIN tuning_source ON tuning_source.id = tuning_space.source_id" +
        DeviceAndTunerJoins +
        "ORDER BY tuning_run.id ASC");

    std::vector<RunSyncRecord> runs;

    while (statement.Step())
    {
        RunSyncRecord row{};
        row.runId = statement.GetSizeT(0);
        row.guid = statement.GetUuid(1);
        row.sourceFingerprint = static_cast<size_t>(std::stoull(statement.GetText(2)));
        row.parameterFingerprint = static_cast<size_t>(std::stoull(statement.GetText(3)));
        row.spaceFingerprint = static_cast<size_t>(std::stoull(statement.GetText(4)));
        row.outputFormat = static_cast<ktt::OutputFormat>(statement.GetInt(5));
        row.inputData = statement.GetOptionalText(6);
        row.createdAt = statement.GetText(7);
        ReadDeviceAndTunerColumns(statement, 8, row.deviceInfo, row.tuner);

        runs.push_back(std::move(row));
    }

    return runs;
}

std::vector<RunQueryResult> RunRepository::GetRunsBySpaceId(
    sqlite3* connection, size_t spaceId, size_t offset, size_t limit
)
{
    Statement statement(connection,
        "SELECT tuning_run.id, tuning_run.guid, tuning_run.input_data, tuning_run.output_format_id, " +
        DeviceAndTunerColumns +
        " FROM tuning_run" +
        DeviceAndTunerJoins +
        "WHERE tuning_run.space_id = ? "
        "ORDER BY tuning_run.id ASC "
        "LIMIT ? OFFSET ?");

    statement.BindInt64(1, static_cast<int64_t>(spaceId));
    const int64_t limitValue = limit == 0 ? -1 : static_cast<int64_t>(limit);
    const int64_t offsetValue = limit == 0 ? 0 : static_cast<int64_t>(offset);
    statement.BindInt64(2, limitValue);
    statement.BindInt64(3, offsetValue);

    std::vector<RunQueryResult> runs;

    while (statement.Step())
    {
        RunQueryResult row{};
        row.runId = statement.GetSizeT(0);
        row.guid = statement.GetUuid(1);
        row.inputData = statement.GetOptionalText(2);
        row.outputFormat = static_cast<ktt::OutputFormat>(statement.GetInt(3));
        ReadDeviceAndTunerColumns(statement, 4, row.deviceInfo, row.tuner);

        runs.push_back(std::move(row));
    }

    return runs;
}

} // namespace ktt::db
