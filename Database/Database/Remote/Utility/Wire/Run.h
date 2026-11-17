#pragma once

#include <optional>
#include <string>
#include <vector>

#include <Api/Info/DatabaseTuningInfo.h>
#include <Api/Output/KernelResult.h>
#include <Database/Remote/Utility/Wire/Common.h>
#include <Database/Remote/Utility/Wire/TuningInfo.h>
#include <Database/Utility/Uuid.h>
#include <Output/TimeConfiguration/TimeUnit.h>

namespace ktt::db::wire
{

/** @struct ResultIn
 * One result of a saved run (ResultIn in ktt_db.json). The payload is the result serialized by ResultSerialization.
 */
struct ResultIn
{
    Nanoseconds duration{};
    Nanoseconds totalDuration{};
    std::string status{"ok"};
    Configuration configuration;
    json result;

    /** @fn static ResultIn From(const KernelResult& result, OutputFormat format)
     * The payload's durations are written in the current time unit, so call it while holding LockTimeUnit.
     * @param format A format returned by GetWireFormat.
     */
    static ResultIn From(const KernelResult &result, OutputFormat format);
};

/** @struct SaveRunRequest
 * The body of POST /run (SaveRunRequest in ktt_db.json).
 */
struct SaveRunRequest
{
    uuid id{};
    TuningInfo tuningInfo;
    OutputFormat outputFormat{OutputFormat::JSON};
    TimeUnit timeUnit{TimeUnit::Nanoseconds};
    std::vector<ResultIn> results;

    /** @fn static SaveRunRequest From(const uuid& id, const db::TuningInfo& info,
     * const std::vector<KernelResult>& results, OutputFormat format)
     * The run with its successful results; failed results are not sent. The payloads are written in the current time
     * unit, which timeUnit records; LockTimeUnit is held meanwhile, so other database threads cannot switch it.
     * @param id The run's id. Every retry of the request sends the same id, so the server saves the run only once.
     * @param format The format of the payloads; XML is sent as JSON (GetWireFormat).
     */
    static SaveRunRequest From(const uuid &id, const db::TuningInfo &info, const std::vector<KernelResult> &results,
        OutputFormat format);
};

/** @struct SaveRunResponse
 * The response to POST /run (SaveRunResponse in ktt_db.json). Read leniently: the results are saved by then, so a
 * missing or malformed source_has_code is std::nullopt rather than an error.
 */
struct SaveRunResponse
{
    std::optional<bool> sourceHasCode;
};

void to_json(json &target, const ResultIn &result);
void to_json(json &target, const SaveRunRequest &request);
void from_json(const json &source, SaveRunResponse &response);

} // namespace ktt::db::wire
