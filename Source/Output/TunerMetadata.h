#pragma once

#include <string>

#include <Api/Info/DeviceInfo.h>
#include <Api/Info/PlatformInfo.h>
#include <ComputeEngine/ComputeApi.h>
#include <Output/TimeConfiguration/TimeUnit.h>

namespace ktt
{

class TunerMetadata
{
public:
    TunerMetadata() = default;
    explicit TunerMetadata(const ComputeApi api, const PlatformInfo& platformInfo, const DeviceInfo& deviceInfo);

    ComputeApi GetComputeApi() const;
    const std::string& GetPlatformName() const;
    const std::string& GetDeviceName() const;
    const std::string& GetKttVersion() const;
    TimeUnit GetTimeUnit() const;

private:
    ComputeApi m_ComputeApi;
    std::string m_PlatformName;
    std::string m_DeviceName;
    std::string m_KttVersion;
    TimeUnit m_TimeUnit;
};

} // namespace ktt
