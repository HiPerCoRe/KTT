#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <Database/Remote/ResultQuery.h>
#include <Database/Remote/Utility/Wire/Common.h>
#include <Database/Remote/Utility/Wire/Device.h>
#include <Database/Utility/Uuid.h>
#include <Output/TimeConfiguration/TimeUnit.h>

namespace ktt::db::wire
{

/** @struct ResultQueryRequest
 * The body of POST /result/query (ResultQueryRequest in ktt_db.json). The options have no "$", and count and distinct
 * are not sent.
 */
struct ResultQueryRequest
{
    std::string filter;
    std::string orderBy;
    uint64_t top{};
    uint64_t skip{};
    std::string select;

    /** @fn static ResultQueryRequest From(const db::ResultQuery& query)
     * Selects the groups ResultQueryItem is read from: run, device and source, and result when the query includes it.
     * Never tuner or space, which db::ResultQueryItem does not hold.
     */
    static ResultQueryRequest From(const db::ResultQuery &query);
};

/** @struct ResultQueryRun
 * The run group of a result query item (ResultQueryRun in ktt_db.json).
 */
struct ResultQueryRun
{
    uuid id{};
    std::optional<std::string> inputData;
    OutputFormat outputFormat{OutputFormat::JSON};
    TimeUnit timeUnit{TimeUnit::Nanoseconds}; ///< The unit of the durations inside the run's payloads.
};

/** @struct ResultQuerySource
 * The source group of a result query item (ResultQuerySource in ktt_db.json), without the name.
 */
struct ResultQuerySource
{
    size_t fingerprint{};
};

/** @struct ResultQueryItem
 * One result of the result query (ResultQueryItem in ktt_db.json), with the groups ResultQueryRequest::From selects.
 */
struct ResultQueryItem
{
    uint64_t duration{};
    uint64_t totalDuration{};
    Configuration configuration;
    ResultQueryRun run;
    DeviceInfo device;
    ResultQuerySource source;
    std::optional<json> result; ///< The payload, as sent; only with "result" in the selected groups.

    /** @fn db::ResultQueryItem ToResultQueryItem(bool includeResult) const
     * @param includeResult Whether the payload was selected; it is then required, and read in the run's format and
     * time unit, which can differ from the unit active now.
     * @throw KttException If the payload was selected but is missing or not a JSON object.
     */
    db::ResultQueryItem ToResultQueryItem(bool includeResult) const;
};

/** @struct ResultQueryResponse
 * The response of POST /result/query (ResultQueryResponse in ktt_db.json), without the total.
 */
struct ResultQueryResponse
{
    std::vector<ResultQueryItem> items;
};

void to_json(json &target, const ResultQueryRequest &request);
void from_json(const json &source, ResultQueryRun &run);
void from_json(const json &source, ResultQuerySource &group);
void from_json(const json &source, ResultQueryItem &item);
void from_json(const json &source, ResultQueryResponse &response);

} // namespace ktt::db::wire
