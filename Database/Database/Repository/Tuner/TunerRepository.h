#pragma once

#include <cstddef>
#include <optional>
#include <sqlite3.h>
#include <string>

#include <Database/Utility/Statement.h>

namespace ktt::db
{

/** @struct Tuner
 * Represents the tuner (name and version) that produced a tuning run.
 */
struct Tuner
{
    std::optional<size_t> id; ///< Unique database identifier.
    std::string name; ///< Tuner name (e.g., "KTT").
    std::string version; ///< Tuner version (e.g., "2.3.1").

    /** @fn static Tuner FromRow(const Statement& statement)
     * Creates a Tuner from a database query result row.
     * @param statement Statement positioned at a result row with columns (id, name, version).
     * @return Tuner populated from the current row.
     */
    static Tuner FromRow(const Statement& statement);
};

/** @class TunerRepository
 * Data access layer for tuner records in the database.
 */
class TunerRepository
{
public:
    /** @fn static Tuner GetOrCreateTuner(sqlite3* connection, const Tuner& tuner)
     * Gets or creates a tuner record in the database.
     * @param connection SQLite database connection.
     * @param tuner Tuner information to get or create.
     * @return Tuner struct with populated id.
     */
    static Tuner GetOrCreateTuner(sqlite3* connection, const Tuner& tuner);

    /** @fn static size_t CreateTuner(sqlite3* connection, const Tuner& tuner)
     * Creates a new tuner record in the database.
     * @param connection SQLite database connection.
     * @param tuner Tuner information to insert.
     * @return Database ID of the newly created tuner record.
     * @throw KttException If insertion fails.
     */
    static size_t CreateTuner(sqlite3* connection, const Tuner& tuner);

    /** @fn static std::optional<Tuner> GetTuner(sqlite3* connection, const Tuner& tuner)
     * Retrieves a tuner record by name and version.
     * @param connection SQLite database connection.
     * @param tuner Tuner information to search for.
     * @return Optional Tuner if found, std::nullopt otherwise.
     * @throw KttException If query fails.
     */
    static std::optional<Tuner> GetTuner(sqlite3* connection, const Tuner& tuner);
};

} // namespace ktt::db
