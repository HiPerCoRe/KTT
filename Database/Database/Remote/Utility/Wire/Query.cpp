#include <Database/Remote/Utility/Wire/Query.h>
#include <Database/Utility/ResultSerialization.h>
#include <Database/Utility/TimeUnitGuard.h>

namespace ktt::db::wire
{

namespace
{

// The groups read besides the core fields: run (id, input data, and the format and time unit of the payload), device
// and source.
const std::string SelectGroups = "run,device,source";

uint64_t DurationFromJson(const json &source, const char *key)
{
    const json &value = source.at(key);

    // The parser stores every non-negative integer as unsigned, so anything else is negative or not an integer.
    if (!value.is_number_unsigned())
        throw KttException(std::string("\"") + key + "\" in the response is not a non-negative integer");

    return value.get<uint64_t>();
}

} // namespace

ResultQueryRequest ResultQueryRequest::From(const db::ResultQuery &query)
{
    ResultQueryRequest request;
    request.filter = query.filter;
    request.orderBy = query.orderBy;
    request.top = query.top;
    request.skip = query.skip;
    request.select = query.includeResult ? SelectGroups + ",result" : SelectGroups;
    return request;
}

db::ResultQueryItem ResultQueryItem::ToResultQueryItem(const bool includeResult) const
{
    db::ResultQueryItem item;
    item.duration = duration;
    item.totalDuration = totalDuration;
    item.configuration = configuration;
    item.runId = UuidGenerator::ToString(run.id);
    item.inputData = run.inputData;
    item.device = device.ToDeviceInfo();
    item.sourceFingerprint = FingerprintToString(source.fingerprint);

    if (includeResult)
    {
        if (!result.has_value())
            throw KttException("a result in the response has no payload");

        if (!result->is_object())
            throw KttException("a result payload in the response is not a JSON object");

        item.result = ResultSerialization::DeserializeResult(result->dump(), run.outputFormat, run.timeUnit);
    }

    return item;
}

void to_json(json &target, const ResultQueryRequest &request)
{
    target = json{
        {"filter", request.filter},
        {"orderby", request.orderBy},
        {"top", request.top},
        {"skip", request.skip},
        {"select", request.select}};
}

void from_json(const json &source, ResultQueryRun &run)
{
    run.id = UuidGenerator::Parse(source.at("id").get<std::string>());
    run.inputData = OptionalFromJson<std::string>(source, "input_data");
    run.outputFormat = OutputFormatFromString(source.at("output_format").get<std::string>());
    run.timeUnit = TimeUnitFromName(source.at("time_unit").get<std::string>());
}

void from_json(const json &source, ResultQuerySource &group)
{
    group.fingerprint = FingerprintFromString(source.at("fingerprint").get<std::string>());
}

void from_json(const json &source, ResultQueryItem &item)
{
    item.duration = DurationFromJson(source, "duration");
    item.totalDuration = DurationFromJson(source, "total_duration");
    item.configuration = ConfigurationFromJson(source.at("configuration"));
    item.run = source.at("run").get<ResultQueryRun>();
    item.device = source.at("device").get<DeviceInfo>();
    item.source = source.at("source").get<ResultQuerySource>();
    item.result = OptionalFromJson<json>(source, "result");
}

void from_json(const json &source, ResultQueryResponse &response)
{
    response.items = ArrayFromJson<ResultQueryItem>(source, "items");
}

} // namespace ktt::db::wire
