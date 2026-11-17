#include <string>

#include <Database/Remote/Utility/Wire/Result.h>
#include <Database/Utility/ResultSerialization.h>
#include <Database/Utility/TimeUnitGuard.h>

namespace ktt::db::wire
{

KernelResult ResultOut::ToKernelResult() const
{
    return ResultSerialization::DeserializeResult(result.dump(), outputFormat, timeUnit);
}

void to_json(json &target, const BestResultsRequest &request)
{
    target = json{{"tuning_info", request.tuningInfo}, {"limit", request.limit}};
}

void to_json(json &target, const RunResultsRequest &request)
{
    json runIds = json::array();

    for (const auto &id : request.runIds)
        runIds.push_back(UuidGenerator::ToString(id));

    target = json{{"run_ids", std::move(runIds)}, {"limit", request.limit}};
}

void from_json(const json &source, ResultOut &result)
{
    result.outputFormat = OutputFormatFromString(source.at("output_format").get<std::string>());
    result.timeUnit = TimeUnitFromName(source.at("time_unit").get<std::string>());
    result.result = source.at("result");

    if (!result.result.is_object())
        throw KttException("a result payload in the response is not a JSON object");

    result.duration = source.at("duration").get<int64_t>();
}

void from_json(const json &source, ResultList &list)
{
    list.results = ArrayFromJson<ResultOut>(source, "results");
}

} // namespace ktt::db::wire
