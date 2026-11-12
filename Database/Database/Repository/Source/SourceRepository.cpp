#include <string>

#include <Database/Repository/Source/SourceRepository.h>
#include <Database/Utility/Statement.h>

namespace ktt::db
{

Source SourceRepository::CreateSource(sqlite3* connection, const Source& source)
{
    Statement statement(connection, R"(
        INSERT INTO tuning_source (source_fingerprint)
        VALUES (?)
    )");

    statement.BindText(1, std::to_string(source.sourceFingerprint));
    statement.Execute();

    const size_t sourceId = static_cast<size_t>(sqlite3_last_insert_rowid(connection));
    return Source{sourceId, source.sourceFingerprint};
}

std::optional<Source> SourceRepository::GetSource(sqlite3* connection, const size_t sourceFingerprint)
{
    Statement statement(connection, R"(
        SELECT id, source_fingerprint
        FROM tuning_source
        WHERE source_fingerprint = ?
        LIMIT 1
    )");

    statement.BindText(1, std::to_string(sourceFingerprint));

    if (!statement.Step())
        return std::nullopt;

    Source source;
    source.id = statement.GetSizeT(0);
    source.sourceFingerprint = static_cast<size_t>(std::stoull(statement.GetText(1)));
    return source;
}

Source SourceRepository::GetOrCreateSource(sqlite3* connection, Source source)
{
    if (auto existingSource = GetSource(connection, source.sourceFingerprint))
        return *existingSource;

    return CreateSource(connection, source);
}

std::optional<SourceStats> SourceRepository::GetStatsForSource(sqlite3* connection, const size_t sourceFingerprint)
{
    Statement statement(connection, R"(
        SELECT
            (SELECT COUNT(*) FROM tuning_space WHERE source_id = tuning_source.id) AS space_count,
            (SELECT COUNT(DISTINCT device.device_info_id)
                FROM tuning_run
                JOIN tuning_space ON tuning_space.id = tuning_run.space_id
                JOIN device ON device.id = tuning_run.device_id
                WHERE tuning_space.source_id = tuning_source.id) AS device_count,
            (SELECT COUNT(*)
                FROM tuning_run
                JOIN tuning_space ON tuning_space.id = tuning_run.space_id
                WHERE tuning_space.source_id = tuning_source.id) AS run_count,
            (SELECT COUNT(*)
                FROM tuning_result
                JOIN tuning_run ON tuning_run.id = tuning_result.run_id
                JOIN tuning_space ON tuning_space.id = tuning_run.space_id
                WHERE tuning_space.source_id = tuning_source.id) AS result_count
        FROM tuning_source
        WHERE source_fingerprint = ?
        LIMIT 1
    )");

    statement.BindText(1, std::to_string(sourceFingerprint));

    if (!statement.Step())
        return std::nullopt;

    SourceStats stats{};
    stats.spaceCount = statement.GetSizeT(0);
    stats.deviceCount = statement.GetSizeT(1);
    stats.runCount = statement.GetSizeT(2);
    stats.resultCount = statement.GetSizeT(3);
    return stats;
}

} // namespace ktt::db
