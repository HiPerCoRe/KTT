#include <Api/Info/DeviceInfo.h>
#include <Utility/ErrorHandling/Assert.h>

namespace ktt
{

DeviceInfo::DeviceInfo(const DeviceIndex index, const std::string& name) :
    m_Index(index),
    m_Name(name),
    m_DeviceType(DeviceType::CPU),
    m_GlobalMemorySize(0),
    m_LocalMemorySize(0),
    m_MaxConstantBufferSize(0),
    m_MaxWorkGroupSize(0),
    m_MaxComputeUnits(0),
    m_CudaComputeCapabilityMajor(0),
    m_CudaComputeCapabilityMinor(0)
{}

DeviceIndex DeviceInfo::GetIndex() const
{
    return m_Index;
}

const std::string& DeviceInfo::GetName() const
{
    return m_Name;
}

const std::string& DeviceInfo::GetVendor() const
{
    return m_Vendor;
}

const std::string& DeviceInfo::GetExtensions() const
{
    return m_Extensions;
}

DeviceType DeviceInfo::GetDeviceType() const
{
    return m_DeviceType;
}

std::string DeviceInfo::GetDeviceTypeString() const
{
    switch (m_DeviceType)
    {
    case ktt::DeviceType::CPU:
        return "CPU";
    case ktt::DeviceType::GPU:
        return "GPU";
    case ktt::DeviceType::Custom:
        return "Custom";
    default:
        KttError("Unhandled device type value");
        return "";
    }
}

uint64_t DeviceInfo::GetGlobalMemorySize() const
{
    return m_GlobalMemorySize;
}

uint64_t DeviceInfo::GetLocalMemorySize() const
{
    return m_LocalMemorySize;
}

uint64_t DeviceInfo::GetMaxConstantBufferSize() const
{
    return m_MaxConstantBufferSize;
}

uint64_t DeviceInfo::GetMaxWorkGroupSize() const
{
    return m_MaxWorkGroupSize;
}

uint32_t DeviceInfo::GetMaxComputeUnits() const
{
    return m_MaxComputeUnits;
}

unsigned int DeviceInfo::GetCUDAComputeCapabilityMajor() const
{
    return m_CudaComputeCapabilityMajor;
}

unsigned int DeviceInfo::GetCUDAComputeCapabilityMinor() const
{
    return m_CudaComputeCapabilityMinor;
}

std::string DeviceInfo::GetString() const
{
    std::string result;

    result += "Information about device with index: " + std::to_string(m_Index) + "\n";
    result += "Name: " + m_Name + "\n";
    result += "Vendor: " + m_Vendor + "\n";
    result += "Device type: " + GetDeviceTypeString() + "\n";
    result += "Global memory size: " + std::to_string(m_GlobalMemorySize) + "\n";
    result += "Local memory size: " + std::to_string(m_LocalMemorySize) + "\n";
    result += "Maximum constant buffer size: " + std::to_string(m_MaxConstantBufferSize) + "\n";
    result += "Maximum work-group size: " + std::to_string(m_MaxWorkGroupSize) + "\n";
    result += "Maximum parallel compute units: " + std::to_string(m_MaxComputeUnits) + "\n";
    result += "Extensions: " + m_Extensions + "\n";
    result += "CUDA compute capability: " + std::to_string(m_CudaComputeCapabilityMajor) + "." + std::to_string(m_CudaComputeCapabilityMinor) + "\n";

    return result;
}

void DeviceInfo::SetVendor(const std::string& vendor)
{
    m_Vendor = vendor;
}

void DeviceInfo::SetExtensions(const std::string& extensions)
{
    m_Extensions = extensions;
}

void DeviceInfo::SetDeviceType(const DeviceType deviceType)
{
    m_DeviceType = deviceType;
}

void DeviceInfo::SetGlobalMemorySize(const uint64_t globalMemorySize)
{
    m_GlobalMemorySize = globalMemorySize;
}

void DeviceInfo::SetLocalMemorySize(const uint64_t localMemorySize)
{
    m_LocalMemorySize = localMemorySize;
}

void DeviceInfo::SetMaxConstantBufferSize(const uint64_t maxConstantBufferSize)
{
    m_MaxConstantBufferSize = maxConstantBufferSize;
}

void DeviceInfo::SetMaxWorkGroupSize(const uint64_t maxWorkGroupSize)
{
    m_MaxWorkGroupSize = maxWorkGroupSize;
}

void DeviceInfo::SetMaxComputeUnits(const uint32_t maxComputeUnits)
{
    m_MaxComputeUnits = maxComputeUnits;
}

void DeviceInfo::SetCUDAComputeCapabilityMajor(const unsigned int cudaComputeCapabilityMajor)
{
    m_CudaComputeCapabilityMajor = cudaComputeCapabilityMajor;
}

void DeviceInfo::SetCUDAComputeCapabilityMinor(const unsigned int cudaComputeCapabilityMinor)
{
    m_CudaComputeCapabilityMinor = cudaComputeCapabilityMinor;
}

} // namespace ktt
