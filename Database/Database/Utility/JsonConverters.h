#pragma once

#include <json.hpp>

namespace ktt::db
{
struct SourceStats;

/** @fn void to_json(nlohmann::json& j, const SourceStats& stats)
 * Serializes a SourceStats struct to JSON format.
 * @param j Reference to the JSON object to populate.
 * @param stats The SourceStats struct to serialize.
 */
void to_json(nlohmann::json& j, const SourceStats& stats);

/** @fn void from_json(const nlohmann::json& j, SourceStats& stats)
 * Deserializes JSON data into a SourceStats struct.
 * @param j The JSON object to deserialize.
 * @param stats Reference to the SourceStats struct to populate.
 */
void from_json(const nlohmann::json& j, SourceStats& stats);

} // namespace ktt::db
