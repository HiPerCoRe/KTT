#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include <Api/Info/DatabaseTuningInfo.h>
#include <Database/Remote/Utility/Wire/Common.h>

namespace ktt::db::wire
{

/** @struct DeviceInfo
 * The device of a run (DeviceInfo in ktt_db.json). Sent in a TuningInfo, read from a run and from a result query.
 */
struct DeviceInfo
{
    std::string name;
    std::string vendor;
    std::string type;
    ComputeApi computeApi{ComputeApi::Cpp};
    std::optional<std::string> deviceIdentifier;
    std::optional<std::string> driverVersion;
    std::optional<std::string> extensions;
    std::optional<uint32_t> cudaComputeCapabilityMajor;
    std::optional<uint32_t> cudaComputeCapabilityMinor;

    /** @fn static DeviceInfo From(const db::DeviceInfo& device)
     * The local Database stores an unknown driver as ""; the server's "unknown" is null, so "" is sent as null.
     */
    static DeviceInfo From(const db::DeviceInfo &device);

    /** @fn db::DeviceInfo ToDeviceInfo() const
     * The device as the local RunRepository reads it: an empty identifier or extensions string is no value, and an
     * unknown driver version is "".
     */
    db::DeviceInfo ToDeviceInfo() const;
};

void to_json(json &target, const DeviceInfo &device);
void from_json(const json &source, DeviceInfo &device);

} // namespace ktt::db::wire
