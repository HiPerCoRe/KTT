#include <cstdint>

#include <Database/Remote/Utility/Wire/Source.h>

namespace ktt::db::wire
{

namespace
{

size_t CountFromJson(const json &source, const char *key)
{
    const auto count = source.at(key).get<int64_t>();

    if (count < 0)
        throw KttException(std::string("\"") + key + "\" in the response is negative");

    return static_cast<size_t>(count);
}

} // namespace

SourceStats SourceStatsResponse::ToSourceStats() const
{
    SourceStats stats;
    stats.spaceCount = spaceCount;
    stats.deviceCount = deviceCount;
    stats.runCount = runCount;
    stats.resultCount = resultCount;
    stats.name = std::nullopt;
    return stats;
}

SaveSourceCodeRequest SaveSourceCodeRequest::From(const std::vector<SourceDefinition> &definitions)
{
    SaveSourceCodeRequest request;
    request.definitions.reserve(definitions.size());

    for (const auto &definition : definitions)
        request.definitions.push_back({definition.name, definition.source});

    return request;
}

void from_json(const json &source, SourceStatsResponse &response)
{
    response.spaceCount = CountFromJson(source, "space_count");
    response.deviceCount = CountFromJson(source, "device_count");
    response.runCount = CountFromJson(source, "run_count");
    response.resultCount = CountFromJson(source, "result_count");
}

void to_json(json &target, const SourceDefinitionIn &definition)
{
    target = json{{"name", definition.name}, {"source", definition.source}};
}

void to_json(json &target, const SaveSourceCodeRequest &request)
{
    target = json{{"definitions", request.definitions}};
}

} // namespace ktt::db::wire
