#include <Database/Local/Database.h>
#include <Database/Local/Utility/JsonConverters.h>

namespace ktt::db
{

void to_json(nlohmann::json& j, const SourceStats& stats)
{
    j = nlohmann::json{
        {"spaceCount", stats.spaceCount},
        {"deviceCount", stats.deviceCount},
        {"runCount", stats.runCount},
        {"resultCount", stats.resultCount},
        {"name", stats.name ? nlohmann::json(*stats.name) : nlohmann::json(nullptr)}
    };
}

void from_json(const nlohmann::json& j, SourceStats& stats)
{
    j.at("spaceCount").get_to(stats.spaceCount);
    j.at("deviceCount").get_to(stats.deviceCount);
    j.at("runCount").get_to(stats.runCount);
    j.at("resultCount").get_to(stats.resultCount);

    if (const auto name = j.find("name"); name != j.end() && !name->is_null())
        stats.name = name->get<std::string>();
    else
        stats.name = std::nullopt;
}

} // namespace ktt::db
