#pragma once

#include <memory>
#include <optional>
#include <sqlite3.h>

#include <Database/Database.h>

namespace ktt::db
{
/** @struct Source
 * Represents a kernel source code fingerprint record.
 */
struct Source
{
    std::optional<size_t> id; ///< Unique database identifier.
    size_t sourceFingerprint; ///< Hash fingerprint of the source code.
};

/** @class SourceRepository
 * Data access layer for tuning sources in the database.
 * Manages kernel source code fingerprints and retrieves statistics about sources.
 */
class SourceRepository
{
public:
    /** @fn static std::optional<Source> GetSource(sqlite3* connection, size_t sourceFingerprint)
     * Retrieves a source by its fingerprint hash.
     * @param connection SQLite database connection.
     * @param sourceFingerprint The source code fingerprint hash to search for.
     * @return Optional Source with populated id field, or std::nullopt if not found.
     * @throw KttException If query fails.
     */
    static std::optional<Source> GetSource(sqlite3* connection, size_t sourceFingerprint);

    /** @fn static Source CreateSource(sqlite3* connection, const Source& source)
     * Creates a new source record in the database.
     * @param connection SQLite database connection.
     * @param source Source information to insert (id field should be empty).
     * @return Source struct with populated id field.
     * @throw KttException If insertion fails.
     */
    static Source CreateSource(sqlite3* connection, const Source& source);

    /** @fn static Source GetOrCreateSource(sqlite3* connection, Source source)
     * Gets or creates a source record in the database.
     * Attempts to retrieve existing source by fingerprint, creates new if not found.
     * @param connection SQLite database connection.
     * @param source Source information to get or create.
     * @return Source struct with populated id field.
     * @throw KttException If operation fails.
     */
    static Source GetOrCreateSource(sqlite3* connection, Source source);

    /** @fn static std::optional<SourceStats> GetStatsForSource(sqlite3* connection, size_t sourceFingerprint)
     * Retrieves statistics for a source.
     * Computes counts of tuning spaces, devices, runs, and results associated with a source.
     * @param connection SQLite database connection.
     * @param sourceFingerprint The source fingerprint to get statistics for.
     * @return Optional SourceStats containing counts, or std::nullopt if source not found.
     * @throw KttException If query fails.
     */
    static std::optional<SourceStats> GetStatsForSource(sqlite3* connection, size_t sourceFingerprint);
};

} // namespace ktt::db
