#include <Database/Database.h>
#include <Database/Utility/JsonConverters.h>

namespace ktt::db
{

void to_json(nlohmann::json& j, const SourceStats& stats)
{
    j = nlohmann::json{
        {"spaceCount", stats.spaceCount},
        {"deviceCount", stats.deviceCount},
        {"runCount", stats.runCount},
        {"resultCount", stats.resultCount}
    };
}

void from_json(const nlohmann::json& j, SourceStats& stats)
{
    j.at("spaceCount").get_to(stats.spaceCount);
    j.at("deviceCount").get_to(stats.deviceCount);
    j.at("runCount").get_to(stats.runCount);
    j.at("resultCount").get_to(stats.resultCount);
}

} // namespace ktt::db
