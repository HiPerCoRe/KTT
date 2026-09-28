#pragma once

#include <cstddef>
#include <optional>
#include <sqlite3.h>
#include <vector>

#include <Api/Info/DatabaseTuningInfo.h>
#include <Database/Utility/Uuid.h>
#include <Output/OutputFormat.h>

namespace ktt::db
{

/** @struct Run
 * Represents a kernel tuning run linking a device to a parameter space.
 */
struct Run
{
    std::optional<size_t> id; ///< Unique database identifier.
    size_t spaceId; ///< Reference to tuning_space.
    size_t deviceId; ///< Reference to device.
    size_t tunerId; ///< Reference to tuner.
    ktt::OutputFormat outputFormat{}; ///< Output format used for the run.
    std::optional<std::string> inputData; ///< Optional input data for the run.
};

/** @struct RunQueryResult
 * Complete run information retrieved from a database query.
 */
struct RunQueryResult
{
    size_t runId{}; ///< Database ID of the run.
    uuid guid{}; ///< Globally unique identifier of the run.
    std::optional<std::string> inputData; ///< Input data associated with the run.
    DeviceInfo deviceInfo{}; ///< Device details.
    TunerInfo tuner{}; ///< Tuner that produced the run.
    ktt::OutputFormat outputFormat{}; ///< Output format used for the run.
};

/** @struct RunSyncRecord
 * Self-contained snapshot of a run together with the source, space and device data needed to
 * recreate it in another database. Produced by RunRepository::GetRuns and consumed by the sync logic.
 */
struct RunSyncRecord
{
    size_t runId{}; ///< Database ID of the run in the originating database (used to fetch its results).
    uuid guid{}; ///< Globally unique identifier of the run, used to detect duplicates across databases.
    size_t sourceFingerprint{}; ///< Fingerprint of the owning tuning source.
    size_t parameterFingerprint{}; ///< Parameter fingerprint of the owning tuning space.
    size_t spaceFingerprint{}; ///< Space fingerprint of the owning tuning space.
    DeviceInfo deviceInfo{}; ///< Device the run was executed on.
    TunerInfo tuner{}; ///< Tuner that produced the run.
    ktt::OutputFormat outputFormat{}; ///< Output format the run's results are serialized in.
    std::optional<std::string> inputData; ///< Optional input data associated with the run.
    std::string createdAt; ///< Original creation timestamp, preserved when copying the run.
};

/** @class RunRepository
 * Data access layer for tuning runs in the database.
 * Manages creation and retrieval of tuning run records that link devices to parameter spaces.
 */
class RunRepository
{
public:
    /** @fn static size_t CreateRun(sqlite3* connection, const Run& run)
     * Creates a new run record in the database.
     * @param connection SQLite database connection.
     * @param run Run information to insert.
     * @return Database ID of the newly created run.
     * @throw KttException If insertion fails.
     */
    static size_t CreateRun(sqlite3* connection, const Run& run);

    /** @fn static size_t CreateRunWithGuid(sqlite3* connection, const Run& run, const uuid& guid, const std::string&
     * createdAt) Creates a new run record using a caller-supplied GUID and creation timestamp instead of generating
     * them. Used when copying a run from another database so its identity and creation time are preserved.
     * @param connection SQLite database connection.
     * @param run Run information to insert.
     * @param guid GUID to store for the run (preserved from the originating database).
     * @param createdAt Creation timestamp to store for the run (preserved from the originating database).
     * @return Database ID of the newly created run.
     * @throw KttException If insertion fails.
     */
    static size_t CreateRunWithGuid(
        sqlite3* connection,
        const Run& run,
        const uuid& guid,
        const std::string& createdAt
    );

    /** @fn static bool RunExists(sqlite3* connection, const uuid& guid)
     * Checks whether a run with the given GUID already exists in the database.
     * @param connection SQLite database connection.
     * @param guid GUID to look for.
     * @return True if a run with the GUID exists, false otherwise.
     * @throw KttException If the query fails.
     */
    static bool RunExists(sqlite3* connection, const uuid& guid);

    /** @fn static std::vector<RunSyncRecord> GetRuns(sqlite3* connection, size_t offset, size_t limit)
     * Retrieves runs in the database together with the source, space and device data required to recreate them
     * elsewhere, with pagination support. Intended for copying runs between databases in batches.
     * @param connection SQLite database connection.
     * @param offset Number of runs to skip (for pagination).
     * @param limit Maximum number of runs to return. If 0, returns all runs.
     * @return Vector of RunSyncRecord objects, ordered by run ID.
     * @throw KttException If the query fails.
     */
    static std::vector<RunSyncRecord> GetRuns(sqlite3* connection, size_t offset, size_t limit);

    /** @fn static std::vector<RunQueryResult> GetRunsBySpaceId(sqlite3* connection, size_t spaceId, size_t offset,
     * size_t limit) Retrieves runs for a tuning space with pagination support. Useful for processing large result sets
     * in batches.
     * @param connection SQLite database connection.
     * @param spaceId Database ID of the tuning space.
     * @param offset Number of results to skip (for pagination).
     * @param limit Maximum number of results to return. If 0, returns all results.
     * @return Vector of RunQueryResult objects, ordered by run ID.
     * @throw KttException If query fails.
     */
    static std::vector<RunQueryResult> GetRunsBySpaceId(
        sqlite3* connection,
        size_t spaceId,
        size_t offset,
        size_t limit
    );
};

} // namespace ktt::db
