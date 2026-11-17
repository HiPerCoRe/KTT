#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <Database/Remote/Utility/Wire/Common.h>
#include <Database/Remote/Utility/Wire/Device.h>
#include <Database/Utility/Uuid.h>

namespace ktt::db::wire
{

/** @struct FindSpaceResponse
 * The response of GET /source/{fingerprint}/space (FindSpaceResponse in ktt_db.json).
 */
struct FindSpaceResponse
{
    int64_t spaceId{};
};

/** @struct SpaceRun
 * A run of a space with what GetBestResults' predicates need (SpaceRun in ktt_db.json).
 */
struct SpaceRun
{
    uuid runId{};
    DeviceInfo device;
    std::optional<std::string> inputData;
};

/** @struct SpaceRunsResponse
 * One page of GET /source/{fingerprint}/space/{id}/run (SpaceRunsResponse in ktt_db.json).
 */
struct SpaceRunsResponse
{
    std::vector<SpaceRun> runs;
};

void from_json(const json &source, FindSpaceResponse &response);
void from_json(const json &source, SpaceRun &run);
void from_json(const json &source, SpaceRunsResponse &response);

} // namespace ktt::db::wire
