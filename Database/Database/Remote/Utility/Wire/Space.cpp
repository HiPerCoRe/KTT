#include <Database/Remote/Utility/Wire/Space.h>

namespace ktt::db::wire
{

void from_json(const json &source, FindSpaceResponse &response)
{
    response.spaceId = source.at("space_id").get<int64_t>();
}

void from_json(const json &source, SpaceRun &run)
{
    run.runId = UuidGenerator::Parse(source.at("run_id").get<std::string>());
    run.device = source.at("device").get<DeviceInfo>();
    run.inputData = OptionalFromJson<std::string>(source, "input_data");
}

void from_json(const json &source, SpaceRunsResponse &response)
{
    response.runs = ArrayFromJson<SpaceRun>(source, "runs");
}

} // namespace ktt::db::wire
