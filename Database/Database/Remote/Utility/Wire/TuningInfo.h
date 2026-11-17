#pragma once

#include <cstddef>
#include <optional>
#include <string>

#include <Api/Info/DatabaseTuningInfo.h>
#include <Database/Remote/Utility/Wire/Common.h>
#include <Database/Remote/Utility/Wire/Device.h>

namespace ktt::db::wire
{

/** @struct TunerInfo
 * The tuner that produced a run (TunerInfo in ktt_db.json).
 */
struct TunerInfo
{
    std::string name;
    std::string version;

    static TunerInfo From(const db::TunerInfo &tuner);
};

/** @struct TuningSpaceInfo
 * The fingerprints and names of a tuning space (TuningSpaceInfo in ktt_db.json). The fingerprints are sent as decimal
 * strings (FingerprintToString).
 */
struct TuningSpaceInfo
{
    size_t sourceFingerprint{};
    size_t parameterFingerprint{};
    size_t spaceFingerprint{};
    std::optional<std::string> sourceName;
    std::optional<std::string> spaceName;

    static TuningSpaceInfo From(const db::TuningSpaceInfo &space);
};

/** @struct TuningInfo
 * What identifies a run (TuningInfo in ktt_db.json): sent with POST /run and POST /result/best.
 */
struct TuningInfo
{
    DeviceInfo device;
    TunerInfo tuner;
    std::optional<std::string> inputData;
    TuningSpaceInfo spaceInfo;

    static TuningInfo From(const db::TuningInfo &info);
};

void to_json(json &target, const TunerInfo &tuner);
void to_json(json &target, const TuningSpaceInfo &space);
void to_json(json &target, const TuningInfo &info);

} // namespace ktt::db::wire
