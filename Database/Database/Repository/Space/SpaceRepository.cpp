#include <string>

#include <Database/Repository/Space/SpaceRepository.h>
#include <Database/Utility/Statement.h>

namespace ktt::db
{

std::optional<Space> SpaceRepository::GetSpace(sqlite3* connection, Space space)
{
    Statement statement(connection, R"(
        SELECT id, source_id, space_fingerprint, parameter_fingerprint
        FROM tuning_space
        WHERE source_id = ? AND space_fingerprint = ? AND parameter_fingerprint = ?
        LIMIT 1
    )");

    statement.BindInt64(1, static_cast<int64_t>(space.sourceId));
    statement.BindText(2, std::to_string(space.spaceFingerprint));
    statement.BindText(3, std::to_string(space.parameterFingerprint));

    if (!statement.Step())
        return std::nullopt;

    Space spaceResult;
    spaceResult.id = statement.GetSizeT(0);
    spaceResult.sourceId = statement.GetSizeT(1);
    spaceResult.spaceFingerprint = static_cast<size_t>(std::stoull(statement.GetText(2)));
    spaceResult.parameterFingerprint = static_cast<size_t>(std::stoull(statement.GetText(3)));
    return spaceResult;
}

Space SpaceRepository::CreateSpace(sqlite3* connection, Space space)
{
    Statement statement(connection, R"(
        INSERT INTO tuning_space (source_id, space_fingerprint, parameter_fingerprint)
        VALUES (?, ?, ?)
    )");

    statement.BindInt64(1, static_cast<int64_t>(space.sourceId));
    statement.BindText(2, std::to_string(space.spaceFingerprint));
    statement.BindText(3, std::to_string(space.parameterFingerprint));
    statement.Execute();

    space.id = static_cast<size_t>(sqlite3_last_insert_rowid(connection));
    return space;
}

Space SpaceRepository::GetOrCreateSpace(sqlite3* connection, Space space)
{
    if (auto existing = GetSpace(connection, space))
        return *existing;

    return CreateSpace(connection, space);
}

} // namespace ktt::db
