#pragma once

#include <memory>
#include <optional>
#include <sqlite3.h>

namespace ktt::db
{

/** @struct Space
 * Represents a kernel parameter search space linked to a source.
 */
struct Space
{
    std::optional<size_t> id; ///< Unique database identifier.
    size_t sourceId; ///< Reference to tuning_source.
    size_t parameterFingerprint; ///< Hash of parameter configuration.
    size_t spaceFingerprint; ///< Hash of search space definition.
};

/** @class SpaceRepository
 * Data access layer for tuning search spaces in the database.
 * Manages tuning spaces that represent parameter configurations linked to kernel sources.
 */
class SpaceRepository
{
public:
    /** @fn static std::optional<Space> GetSpace(sqlite3* connection, Space space)
     * Retrieves a space from the database by its fingerprints and source.
     * @param connection SQLite database connection.
     * @param space Space information to search for (id field is ignored).
     * @return Optional Space with populated id field, or std::nullopt if not found.
     * @throw KttException If query fails.
     */
    static std::optional<Space> GetSpace(sqlite3* connection, Space space);

    /** @fn static Space CreateSpace(sqlite3* connection, Space space)
     * Creates a new tuning space record in the database.
     * @param connection SQLite database connection.
     * @param space Space information to insert (id field should be empty).
     * @return Space struct with populated id field.
     * @throw KttException If insertion fails.
     */
    static Space CreateSpace(sqlite3* connection, Space space);

    /** @fn static Space GetOrCreateSpace(sqlite3* connection, Space space)
     * Gets or creates a tuning space record in the database.
     * Attempts to retrieve existing space by fingerprints, creates new if not found.
     * @param connection SQLite database connection.
     * @param space Space information to get or create.
     * @return Space struct with populated id field.
     * @throw KttException If operation fails.
     */
    static Space GetOrCreateSpace(sqlite3* connection, Space space);
};

} // namespace ktt::db
