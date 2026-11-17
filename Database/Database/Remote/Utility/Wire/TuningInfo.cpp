#include <Database/Remote/Utility/Wire/TuningInfo.h>

namespace ktt::db::wire
{

TunerInfo TunerInfo::From(const db::TunerInfo &tuner)
{
    return TunerInfo{tuner.name, tuner.version};
}

TuningSpaceInfo TuningSpaceInfo::From(const db::TuningSpaceInfo &space)
{
    TuningSpaceInfo result;
    result.sourceFingerprint = space.sourceFingerprint;
    result.parameterFingerprint = space.parameterFingerprint;
    result.spaceFingerprint = space.spaceFingerprint;
    result.sourceName = space.sourceName;
    result.spaceName = space.spaceName;
    return result;
}

TuningInfo TuningInfo::From(const db::TuningInfo &info)
{
    TuningInfo result;
    result.device = DeviceInfo::From(info.device);
    result.tuner = TunerInfo::From(info.tuner);
    result.inputData = info.inputData;
    result.spaceInfo = TuningSpaceInfo::From(info.spaceInfo);
    return result;
}

void to_json(json &target, const TunerInfo &tuner)
{
    target = json{{"name", tuner.name}, {"version", tuner.version}};
}

void to_json(json &target, const TuningSpaceInfo &space)
{
    target = json{
        {"source_fingerprint", FingerprintToString(space.sourceFingerprint)},
        {"parameter_fingerprint", FingerprintToString(space.parameterFingerprint)},
        {"space_fingerprint", FingerprintToString(space.spaceFingerprint)},
        {"source_name", OptionalToJson(space.sourceName)},
        {"space_name", OptionalToJson(space.spaceName)}};
}

void to_json(json &target, const TuningInfo &info)
{
    target = json{
        {"device", info.device},
        {"tuner", info.tuner},
        {"input_data", OptionalToJson(info.inputData)},
        {"space_info", info.spaceInfo}};
}

} // namespace ktt::db::wire
