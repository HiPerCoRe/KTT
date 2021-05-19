#include <chrono>
#include <sstream>
#include <date.h>

#include <Api/Info/DeviceInfo.h>
#include <Api/Info/PlatformInfo.h>
#include <ComputeEngine/ComputeEngine.h>
#include <Output/TimeConfiguration/TimeConfiguration.h>
#include <Output/TunerMetadata.h>
#include <KttPlatform.h>

namespace ktt
{

TunerMetadata::TunerMetadata(const ComputeEngine& engine) :
    m_ComputeApi(engine.GetComputeApi()),
    m_GlobalSizeType(engine.GetGlobalSizeType()),
    m_PlatformName(engine.GetCurrentPlatformInfo().GetName()),
    m_DeviceName(engine.GetCurrentDeviceInfo().GetName())
{
    m_KttVersion = GetKttVersionString();
    m_TimeUnit = TimeConfiguration::GetInstance().GetTimeUnit();

    using namespace date;

    const auto now = std::chrono::system_clock::now();
    const auto today = date::floor<days>(now);

    std::stringstream stream;
    stream << today << ' ' << make_time(now - today) << " UTC";
    m_Timestamp = stream.str();
}

void TunerMetadata::SetComputeApi(const ComputeApi api)
{
    m_ComputeApi = api;
}

void TunerMetadata::SetGlobalSizeType(const GlobalSizeType sizeType)
{
    m_GlobalSizeType = sizeType;
}

void TunerMetadata::SetPlatformName(const std::string& name)
{
    m_PlatformName = name;
}

void TunerMetadata::SetDeviceName(const std::string& name)
{
    m_DeviceName = name;
}

void TunerMetadata::SetKttVersion(const std::string& version)
{
    m_KttVersion = version;
}

void TunerMetadata::SetTimestamp(const std::string& timestamp)
{
    m_Timestamp = timestamp;
}

void TunerMetadata::SetTimeUnit(const TimeUnit unit)
{
    m_TimeUnit = unit;
}

ComputeApi TunerMetadata::GetComputeApi() const
{
    return m_ComputeApi;
}

GlobalSizeType TunerMetadata::GetGlobalSizeType() const
{
    return m_GlobalSizeType;
}

const std::string& TunerMetadata::GetPlatformName() const
{
    return m_PlatformName;
}

const std::string& TunerMetadata::GetDeviceName() const
{
    return m_DeviceName;
}

const std::string& TunerMetadata::GetKttVersion() const
{
    return m_KttVersion;
}

const std::string& TunerMetadata::GetTimestamp() const
{
    return m_Timestamp;
}

TimeUnit TunerMetadata::GetTimeUnit() const
{
    return m_TimeUnit;
}

} // namespace ktt
