#include <utility>

#include <Database/Remote/Utility/Wire/Device.h>

namespace ktt::db::wire
{

namespace
{

std::optional<std::string> NonEmpty(const std::optional<std::string> &text)
{
    return text.has_value() && !text->empty() ? text : std::nullopt;
}

} // namespace

DeviceInfo DeviceInfo::From(const db::DeviceInfo &device)
{
    DeviceInfo result;
    result.name = device.name;
    result.vendor = device.vendor;
    result.type = device.type;
    result.computeApi = device.computeApi;
    result.deviceIdentifier = device.deviceIdentifier;
    result.driverVersion = device.driverVersion.empty() ? std::nullopt : std::optional<std::string>(device.driverVersion);
    result.extensions = device.extensions;
    result.cudaComputeCapabilityMajor = device.cudaComputeCapabilityMajor;
    result.cudaComputeCapabilityMinor = device.cudaComputeCapabilityMinor;
    return result;
}

db::DeviceInfo DeviceInfo::ToDeviceInfo() const
{
    db::DeviceInfo device;
    device.name = name;
    device.vendor = vendor;
    device.type = type;
    device.computeApi = computeApi;
    device.deviceIdentifier = NonEmpty(deviceIdentifier);
    device.driverVersion = driverVersion.value_or("");
    device.extensions = NonEmpty(extensions);
    device.cudaComputeCapabilityMajor = cudaComputeCapabilityMajor;
    device.cudaComputeCapabilityMinor = cudaComputeCapabilityMinor;
    return device;
}

void to_json(json &target, const DeviceInfo &device)
{
    target = json{
        {"name", device.name},
        {"vendor", device.vendor},
        {"type", device.type},
        {"compute_api", ComputeApiToString(device.computeApi)},
        {"device_identifier", OptionalToJson(device.deviceIdentifier)},
        {"driver_version", OptionalToJson(device.driverVersion)},
        {"extensions", OptionalToJson(device.extensions)},
        {"cuda_compute_capability_major", OptionalToJson(device.cudaComputeCapabilityMajor)},
        {"cuda_compute_capability_minor", OptionalToJson(device.cudaComputeCapabilityMinor)}};
}

void from_json(const json &source, DeviceInfo &device)
{
    device.name = source.at("name").get<std::string>();
    device.vendor = source.at("vendor").get<std::string>();
    device.type = source.at("type").get<std::string>();
    device.computeApi = ComputeApiFromString(source.at("compute_api").get<std::string>());
    device.deviceIdentifier = OptionalFromJson<std::string>(source, "device_identifier");
    device.driverVersion = OptionalFromJson<std::string>(source, "driver_version");
    device.extensions = OptionalFromJson<std::string>(source, "extensions");
    device.cudaComputeCapabilityMajor = OptionalFromJson<uint32_t>(source, "cuda_compute_capability_major");
    device.cudaComputeCapabilityMinor = OptionalFromJson<uint32_t>(source, "cuda_compute_capability_minor");
}

} // namespace ktt::db::wire
