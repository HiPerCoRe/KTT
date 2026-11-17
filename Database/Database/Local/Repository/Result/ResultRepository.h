#pragma once
#include <Ktt.h>
#include <memory>
#include <sqlite3.h>
#include <string>
#include <vector>

#include <Database/Local/Repository/Device/DeviceRepository.h>
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

/** @struct StoredResult
 * A loaded kernel result together with its stored kernel duration (tuning_result.duration), the key the queries
 * order by. The stored duration can differ from the loaded result's KernelResult::GetKernelDuration: JSON_T4 keeps
 * only the total duration, and the durations inside the serialized result lose precision below the run's time unit.
 */
struct StoredResult
{
    int64_t duration{}; ///< Stored kernel duration in nanoseconds (tuning_result.duration).
    KernelResult result; ///< The result, deserialized with the time unit stored on its run.
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
     * ordered by the stored kernel duration (fastest first). A run's device matches when it is the same model (name,
     * vendor and type), or, for CUDA only, when it has the same compute capability major and a minor within +-1. Other
     * compute APIs match by model only. Each result is deserialized with the time unit stored on its run.
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

    /** @fn static std::vector<StoredResult> ResultsByRunIds(sqlite3* connection, const std::vector<size_t>& runIds,
     * uint32_t limit) Retrieves kernel results for a list of run IDs, deserialized with the time unit stored on
     * their runs, each together with its stored kernel duration.
     * @param connection SQLite database connection.
     * @param runIds Vector of run IDs to retrieve results for.
     * @param limit Maximum number of results to return.
     * @return Vector of StoredResult objects sorted by the stored kernel duration (StoredResult::duration), or an
     * empty vector when runIds is empty or limit is 0.
     * @throw KttException If query fails.
     */
    static std::vector<StoredResult> ResultsByRunIds(
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
