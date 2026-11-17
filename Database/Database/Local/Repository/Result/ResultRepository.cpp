#include <Database/Local/Repository/Result/ResultRepository.h>
#include <Database/Local/Utility/SqlList.h>
#include <Database/Local/Utility/Statement.h>
#include <Database/Utility/ResultSerialization.h>
#include <Database/Utility/TimeUnitGuard.h>
#include <Output/OutputFormat.h>

namespace ktt::db
{

void ResultRepository::CreateResults(
    sqlite3* connection,
    const size_t runId,
    const std::vector<KernelResult>& results,
    const ktt::OutputFormat format,
    const int indentResultsJson
)
{
    Statement statement(connection, R"(
        INSERT INTO tuning_result (run_id, duration, result)
        VALUES (?, ?, ?)
    )");

    for (const auto& result : results)
    {
        if (result.GetStatus() != ResultStatus::Ok)
            continue;

        statement.BindInt64(1, static_cast<int64_t>(runId));
        statement.BindInt64(2, static_cast<int64_t>(result.GetKernelDuration()));
        statement.BindText(3, ResultSerialization::SerializeResult(result, format, indentResultsJson));
        statement.Execute();
        statement.Reset();
    }
}

std::vector<KernelResult> ResultRepository::SimpleGetBestResults(
    sqlite3* connection, size_t spaceId, const Device& device, uint32_t limit
)
{
    // A run's device matches when it is the same model (name, vendor and type), or, for CUDA only, when it has the
    // same compute capability major and a minor within +-1. Other compute APIs match by model only.
    Statement statement(connection, R"(
        SELECT tuning_result.result, tuning_run.output_format_id, tuning_run.time_unit
        FROM tuning_result
        JOIN tuning_run ON tuning_run.id = tuning_result.run_id
        JOIN device ON device.id = tuning_run.device_id
        JOIN device_info ON device_info.id = device.device_info_id
        JOIN device_api ON device_api.id = device.device_api_id
        WHERE tuning_run.space_id = ?
            AND ((device_info.name = ?
                    AND device_info.vendor = ?
                    AND device_info.type = ?)
                OR (? = 1
                    AND device_api.compute_api_id = ?
                    AND device_api.version_major = ?
                    AND device_api.version_minor BETWEEN ? AND ?))
        ORDER BY tuning_result.duration ASC
        LIMIT ?
    )");

    const bool isCuda = device.computeApi == ComputeApi::CUDA && device.cudaComputeCapabilityMajor.has_value()
        && device.cudaComputeCapabilityMinor.has_value();
    const int major = device.cudaComputeCapabilityMajor.value_or(NoCudaComputeCapability);
    const int minor = device.cudaComputeCapabilityMinor.value_or(NoCudaComputeCapability);

    statement.BindInt64(1, static_cast<int64_t>(spaceId));
    statement.BindText(2, device.name);
    statement.BindText(3, device.vendor);
    statement.BindText(4, device.type);
    statement.BindInt(5, isCuda ? 1 : 0);
    statement.BindInt(6, static_cast<int>(ComputeApi::CUDA));
    statement.BindInt(7, major);
    statement.BindInt(8, minor - 1);
    statement.BindInt(9, minor + 1);
    statement.BindInt(10, static_cast<int>(limit));

    std::vector<KernelResult> results;
    while (statement.Step())
    {
        const auto format = static_cast<ktt::OutputFormat>(statement.GetInt(1));
        const auto timeUnit = TimeUnitFromName(statement.GetText(2));
        results.push_back(ResultSerialization::DeserializeResult(statement.GetText(0), format, timeUnit));
    }

    return results;
}

std::vector<StoredResult> ResultRepository::ResultsByRunIds(
    sqlite3* connection, const std::vector<size_t>& runIds, uint32_t limit
)
{
    if (runIds.empty() || limit == 0)
        return {};

    Statement statement(connection,
        "SELECT tuning_result.result, tuning_run.output_format_id, tuning_run.time_unit, tuning_result.duration "
        "FROM tuning_result "
        "JOIN tuning_run ON tuning_run.id = tuning_result.run_id "
        "WHERE tuning_result.run_id IN " +
        SqlList(runIds.size()) + " ORDER BY tuning_result.duration ASC LIMIT ?");

    int bindIndex = 1;
    for (const auto runId : runIds)
    {
        statement.BindInt64(bindIndex, static_cast<int64_t>(runId));
        ++bindIndex;
    }
    statement.BindInt(bindIndex, static_cast<int>(limit));

    std::vector<StoredResult> results;
    while (statement.Step())
    {
        const auto format = static_cast<ktt::OutputFormat>(statement.GetInt(1));
        const auto timeUnit = TimeUnitFromName(statement.GetText(2));
        results.push_back(
            {statement.GetInt64(3), ResultSerialization::DeserializeResult(statement.GetText(0), format, timeUnit)});
    }

    return results;
}

std::vector<RawResult> ResultRepository::GetRawResultsByRunId(sqlite3* connection, const size_t runId)
{
    Statement statement(connection, R"(
        SELECT duration, result
        FROM tuning_result
        WHERE run_id = ?
        ORDER BY id ASC
    )");

    statement.BindInt64(1, static_cast<int64_t>(runId));

    std::vector<RawResult> results;
    while (statement.Step())
    {
        RawResult row{};
        row.duration = statement.GetInt64(0);
        row.result = statement.GetText(1);
        results.push_back(std::move(row));
    }

    return results;
}

void ResultRepository::InsertRawResults(sqlite3* connection, const size_t runId, const std::vector<RawResult>& results)
{
    Statement statement(connection, R"(
        INSERT INTO tuning_result (run_id, duration, result)
        VALUES (?, ?, ?)
    )");

    for (const auto& result : results)
    {
        statement.BindInt64(1, static_cast<int64_t>(runId));
        statement.BindInt64(2, result.duration);
        statement.BindText(3, result.result);
        statement.Execute();
        statement.Reset();
    }
}

} // namespace ktt::db
