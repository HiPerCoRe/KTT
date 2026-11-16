#pragma once
#include <Ktt.h>
#include <memory>
#include <sqlite3.h>
#include <string>
#include <vector>

#include <Database/Repository/Device/DeviceRepository.h>
#include <Output/OutputFormat.h>

namespace ktt::db
{

/** @struct RawResult
 * A kernel result row as stored in the database, kept in its serialized form.
 * Used to copy results between databases verbatim, without re-serializing them.
 */
struct RawResult
{
    int64_t duration{}; ///< Stored kernel duration.
    std::string result; ///< Serialized result payload, in the owning run's output format.
};

/** @class ResultRepository
 * Data access layer for kernel execution results in the database.
 * Manages storage and retrieval of kernel execution results with performance metrics.
 */
class ResultRepository
{
public:
    /** @fn static void CreateResults(sqlite3* connection, size_t runId, const std::vector<KernelResult>& results,
     * ktt::OutputFormat format, int indentResultsJson) Stores kernel execution results in the database. Inserts
     * successful results (ResultStatus::Ok) serialized in the run's output format.
     * @param connection SQLite database connection.
     * @param runId Database ID of the run these results belong to.
     * @param results Vector of KernelResult objects to store.
     * @param format Output format used to serialize each result.
     * @param indentResultsJson Indentation level for JSON serialization (ignored for XML).
     * @throw KttException If insertion fails.
     */
    static void CreateResults(
        sqlite3* connection,
        size_t runId,
        const std::vector<KernelResult>& results,
        ktt::OutputFormat format,
        int indentResultsJson
    );

    /** @fn static std::vector<KernelResult> SimpleGetBestResults(sqlite3* connection, size_t spaceId, const Device&
     * device, uint32_t limit = 50) Retrieves the best kernel results for a space and device combination. Results are
     * ordered by duration (fastest first). Supports device matching by exact name/vendor/type or by compute API and
     * version.
     * @param connection SQLite database connection.
     * @param spaceId Database ID of the tuning space.
     * @param device Device information to filter by.
     * @param limit Maximum number of results to return (default 50).
     * @return Vector of KernelResult objects sorted by execution time.
     * @throw KttException If query fails.
     */
    static std::vector<KernelResult> SimpleGetBestResults(
        sqlite3* connection, size_t spaceId, const Device& device, uint32_t limit
    );

    /** @fn static std::vector<KernelResult> ResultsByRunIds(sqlite3* connection, const std::vector<size_t>& runIds,
     * uint32_t limit) Retrieves kernel results for a list of run IDs.
     * @param connection SQLite database connection.
     * @param runIds Vector of run IDs to retrieve results for.
     * @param limit Maximum number of results to return.
     * @return Vector of KernelResult objects sorted by execution time.
     * @throw KttException If query fails or runIds is empty.
     */
    static std::vector<KernelResult> ResultsByRunIds(
        sqlite3* connection, const std::vector<size_t>& runIds, uint32_t limit
    );

    /** @fn static std::vector<RawResult> GetRawResultsByRunId(sqlite3* connection, size_t runId)
     * Retrieves the results of a run in their stored, serialized form, without deserializing them.
     * Intended for copying results between databases.
     * @param connection SQLite database connection.
     * @param runId Database ID of the run whose results to retrieve.
     * @return Vector of RawResult objects in their stored order.
     * @throw KttException If the query fails.
     */
    static std::vector<RawResult> GetRawResultsByRunId(sqlite3* connection, size_t runId);

    /** @fn static void InsertRawResults(sqlite3* connection, size_t runId, const std::vector<RawResult>& results)
     * Inserts already-serialized results for a run, copying them verbatim.
     * @param connection SQLite database connection.
     * @param runId Database ID of the run these results belong to.
     * @param results Serialized results to insert.
     * @throw KttException If insertion fails.
     */
    static void InsertRawResults(sqlite3* connection, size_t runId, const std::vector<RawResult>& results);
};

} // namespace ktt::db
