#include <Database/Repository/Tuner/TunerRepository.h>

namespace ktt::db
{

size_t TunerRepository::CreateTuner(sqlite3* connection, const Tuner& tuner)
{
    Statement statement(connection, R"(
        INSERT INTO tuner
        (name, version)
        VALUES (?, ?)
    )");

    statement.BindText(1, tuner.name);
    statement.BindText(2, tuner.version);
    statement.Execute();

    return sqlite3_last_insert_rowid(connection);
}

std::optional<Tuner> TunerRepository::GetTuner(sqlite3* connection, const Tuner& tuner)
{
    Statement statement(connection, R"(
        SELECT id, name, version
        FROM tuner
        WHERE name = ? AND version = ?
        LIMIT 1
    )");

    statement.BindText(1, tuner.name);
    statement.BindText(2, tuner.version);

    if (!statement.Step())
        return std::nullopt;

    return Tuner::FromRow(statement);
}

Tuner TunerRepository::GetOrCreateTuner(sqlite3* connection, const Tuner& tuner)
{
    Tuner output = tuner;

    if (auto existingTuner = GetTuner(connection, tuner))
        output.id = existingTuner->id;
    else
        output.id = CreateTuner(connection, tuner);

    return output;
}

Tuner Tuner::FromRow(const Statement& statement)
{
    return Tuner{
        statement.GetSizeT(0),
        statement.GetText(1),
        statement.GetText(2)
    };
}

} // namespace ktt::db
