#pragma once

#include <cstdint>
#include <vector>

#include <Api/Output/KernelResult.h>
#include <Database/Remote/Utility/Wire/Common.h>
#include <Database/Remote/Utility/Wire/TuningInfo.h>
#include <Database/Utility/Uuid.h>
#include <Output/TimeConfiguration/TimeUnit.h>

namespace ktt::db::wire
{

/** @struct BestResultsRequest
 * The body of POST /result/best (BestResultsRequest in ktt_db.json).
 */
struct BestResultsRequest
{
    TuningInfo tuningInfo;
    uint32_t limit{};
};

/** @struct RunResultsRequest
 * The body of POST /source/{fingerprint}/space/{id}/result (RunResultsRequest in ktt_db.json): the best results of
 * the given runs.
 */
struct RunResultsRequest
{
    std::vector<uuid> runIds;
    uint32_t limit{};
};

/** @struct ResultOut
 * A result read back (ResultOut in ktt_db.json). The payload stays JSON until ToKernelResult, which reads it in the
 * format and time unit of its run.
 */
struct ResultOut
{
    int64_t duration{}; ///< Kernel duration in nanoseconds, what the server orders by.
    OutputFormat outputFormat{OutputFormat::JSON};
    TimeUnit timeUnit{TimeUnit::Nanoseconds};
    json result; ///< The payload, always a JSON object.

    KernelResult ToKernelResult() const;
};

/** @struct ResultList
 * The response of POST /result/best and POST .../space/{id}/result (ResultList in ktt_db.json), in the server's
 * order.
 */
struct ResultList
{
    std::vector<ResultOut> results;
};

void to_json(json &target, const BestResultsRequest &request);
void to_json(json &target, const RunResultsRequest &request);
void from_json(const json &source, ResultOut &result);
void from_json(const json &source, ResultList &list);

} // namespace ktt::db::wire
