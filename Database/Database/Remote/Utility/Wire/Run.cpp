#include <variant>

#include <Database/Remote/Utility/Wire/Run.h>
#include <Database/Utility/ResultSerialization.h>
#include <Database/Utility/TimeUnitGuard.h>
#include <Output/TimeConfiguration/TimeConfiguration.h>

namespace ktt::db::wire
{

namespace
{

ParameterValue ParameterValueFrom(const ParameterPair &pair)
{
    switch (pair.GetValueType())
    {
    case ParameterValueType::Int:
    case ParameterValueType::UnsignedInt:
    case ParameterValueType::Double:
    case ParameterValueType::Bool:
        return pair.GetValue();
    case ParameterValueType::String:
    case ParameterValueType::CompilerParameter:
    default:
        return pair.GetValueString();
    }
}

} // namespace

ResultIn ResultIn::From(const KernelResult &result, const OutputFormat format)
{
    ResultIn wireResult;
    wireResult.duration = result.GetKernelDuration();
    wireResult.totalDuration = result.GetTotalDuration();

    for (const auto &pair : result.GetConfiguration().GetPairs())
        wireResult.configuration[pair.GetName()] = ParameterValueFrom(pair);

    wireResult.result = json::parse(ResultSerialization::SerializeResult(result, format, 0));
    return wireResult;
}

SaveRunRequest SaveRunRequest::From(const uuid &id, const db::TuningInfo &info,
    const std::vector<KernelResult> &results, const OutputFormat format)
{
    SaveRunRequest request;
    request.id = id;
    request.tuningInfo = TuningInfo::From(info);
    request.outputFormat = GetWireFormat(format);

    const auto lock = LockTimeUnit();
    request.timeUnit = TimeConfiguration::GetInstance().GetTimeUnit();

    for (const auto &result : results)
    {
        if (result.GetStatus() == ResultStatus::Ok)
            request.results.push_back(ResultIn::From(result, request.outputFormat));
    }

    return request;
}

void to_json(json &target, const ResultIn &result)
{
    target = json{
        {"duration", result.duration},
        {"total_duration", result.totalDuration},
        {"status", result.status},
        {"configuration", ConfigurationToJson(result.configuration)},
        {"result", result.result}};
}

void to_json(json &target, const SaveRunRequest &request)
{
    target = json{
        {"id", UuidGenerator::ToString(request.id)},
        {"tuning_info", request.tuningInfo},
        {"output_format", OutputFormatToString(request.outputFormat)},
        {"time_unit", TimeUnitToName(request.timeUnit)},
        {"results", request.results}};
}

void from_json(const json &source, SaveRunResponse &response)
{
    const auto hasCode = source.find("source_has_code");
    response.sourceHasCode =
        hasCode != source.end() && hasCode->is_boolean() ? std::optional<bool>(hasCode->get<bool>()) : std::nullopt;
}

} // namespace ktt::db::wire
