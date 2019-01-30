#include <api/device_info.h>

namespace ktt
{

DeviceInfo::DeviceInfo(const DeviceIndex device, const std::string& name) :
    id(device),
    name(name)
{}

DeviceIndex DeviceInfo::getId() const
{
    return id;
}

const std::string& DeviceInfo::getName() const
{
    return name;
}

const std::string& DeviceInfo::getVendor() const
{
    return vendor;
}

const std::string& DeviceInfo::getExtensions() const
{
    return extensions;
}

DeviceType DeviceInfo::getDeviceType() const
{
    return deviceType;
}

std::string DeviceInfo::getDeviceTypeAsString() const
{
    switch (deviceType)
    {
    case ktt::DeviceType::Accelerator:
        return std::string("Accelerator");
    case ktt::DeviceType::CPU:
        return std::string("CPU");
    case ktt::DeviceType::Custom:
        return std::string("Custom");
    case ktt::DeviceType::Default:
        return std::string("Default");
    default:
        return std::string("GPU");
    }
}

uint64_t DeviceInfo::getGlobalMemorySize() const
{
    return globalMemorySize;
}

uint64_t DeviceInfo::getLocalMemorySize() const
{
    return localMemorySize;
}

uint64_t DeviceInfo::getMaxConstantBufferSize() const
{
    return maxConstantBufferSize;
}

uint32_t DeviceInfo::getMaxComputeUnits() const
{
    return maxComputeUnits;
}

size_t DeviceInfo::getMaxWorkGroupSize() const
{
    return maxWorkGroupSize;
}

void DeviceInfo::setVendor(const std::string& vendor)
{
    this->vendor = vendor;
}

void DeviceInfo::setExtensions(const std::string& extensions)
{
    this->extensions = extensions;
}

void DeviceInfo::setDeviceType(const DeviceType deviceType)
{
    this->deviceType = deviceType;
}

void DeviceInfo::setGlobalMemorySize(const uint64_t globalMemorySize)
{
    this->globalMemorySize = globalMemorySize;
}

void DeviceInfo::setLocalMemorySize(const uint64_t localMemorySize)
{
    this->localMemorySize = localMemorySize;
}

void DeviceInfo::setMaxConstantBufferSize(const uint64_t maxConstantBufferSize)
{
    this->maxConstantBufferSize = maxConstantBufferSize;
}

void DeviceInfo::setMaxComputeUnits(const uint32_t maxComputeUnits)
{
    this->maxComputeUnits = maxComputeUnits;
}

void DeviceInfo::setMaxWorkGroupSize(const size_t maxWorkGroupSize)
{
    this->maxWorkGroupSize = maxWorkGroupSize;
}

std::ostream& operator<<(std::ostream& outputTarget, const DeviceInfo& deviceInfo)
{
    outputTarget << "Printing detailed info for device with index: " << deviceInfo.getId() << std::endl;
    outputTarget << "Name: " << deviceInfo.getName() << std::endl;
    outputTarget << "Vendor: " << deviceInfo.getVendor() << std::endl;
    outputTarget << "Device type: " << deviceInfo.getDeviceTypeAsString() << std::endl;
    outputTarget << "Global memory size: " << deviceInfo.getGlobalMemorySize() << std::endl;
    outputTarget << "Local memory size: " << deviceInfo.getLocalMemorySize() << std::endl;
    outputTarget << "Maximum constant buffer size: " << deviceInfo.getMaxConstantBufferSize() << std::endl;
    outputTarget << "Maximum parallel compute units: " << deviceInfo.getMaxComputeUnits() << std::endl;
    outputTarget << "Maximum work-group size: " << deviceInfo.getMaxWorkGroupSize() << std::endl;
    outputTarget << "Extensions: " << deviceInfo.getExtensions() << std::endl;
    return outputTarget;
}

} // namespace ktt
