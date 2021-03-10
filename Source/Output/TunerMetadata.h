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

    void SetComputeApi(const ComputeApi api);
    void SetPlatformName(const std::string& name);
    void SetDeviceName(const std::string& name);
    void SetKttVersion(const std::string& version);
    void SetTimestamp(const std::string& timestamp);
    void SetTimeUnit(const TimeUnit unit);

    ComputeApi GetComputeApi() const;
    const std::string& GetPlatformName() const;
    const std::string& GetDeviceName() const;
    const std::string& GetKttVersion() const;
    const std::string& GetTimestamp() const;
    TimeUnit GetTimeUnit() const;

private:
    ComputeApi m_ComputeApi;
    std::string m_PlatformName;
    std::string m_DeviceName;
    std::string m_KttVersion;
    std::string m_Timestamp;
    TimeUnit m_TimeUnit;
};

} // namespace ktt
