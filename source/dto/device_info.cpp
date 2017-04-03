#include "device_info.h"

namespace ktt
{

DeviceInfo::DeviceInfo(const size_t id, const std::string& name):
    id(id),
    name(name)
{}

size_t DeviceInfo::getId() const
{
    return id;
}

std::string DeviceInfo::getName() const
{
    return name;
}

std::string DeviceInfo::getVendor() const
{
    return vendor;
}

std::string DeviceInfo::getExtensions() const
{
    return extensions;
}

DeviceType DeviceInfo::getDeviceType() const
{
    return deviceType;
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

void DeviceInfo::setDeviceType(const DeviceType& deviceType)
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

std::string deviceTypeToString(const DeviceType& deviceType)
{
    switch (deviceType)
    {
    case ktt::DeviceType::ACCELERATOR:
        return std::string("ACCELERATOR");
    case ktt::DeviceType::CPU:
        return std::string("CPU");
    case ktt::DeviceType::CUSTOM:
        return std::string("CUSTOM");
    case ktt::DeviceType::DEFAULT:
        return std::string("DEFAULT");
    default:
        return std::string("GPU");
    }
}

std::ostream& operator<<(std::ostream& outputTarget, const DeviceInfo& deviceInfo)
{
    outputTarget << "Printing detailed info for device with id: " << deviceInfo.id << std::endl;
    outputTarget << "Name: " << deviceInfo.name << std::endl;
    outputTarget << "Vendor: " << deviceInfo.vendor << std::endl;
    outputTarget << "Device type: " << deviceTypeToString(deviceInfo.deviceType) << std::endl;
    outputTarget << "Global memory size: " << deviceInfo.globalMemorySize << std::endl;
    outputTarget << "Local memory size: " << deviceInfo.localMemorySize << std::endl;
    outputTarget << "Maximum constant buffer size: " << deviceInfo.maxConstantBufferSize << std::endl;
    outputTarget << "Maximum parallel compute units: " << deviceInfo.maxComputeUnits << std::endl;
    outputTarget << "Maximum work-group size: " << deviceInfo.maxWorkGroupSize << std::endl;
    outputTarget << "Extensions: " << deviceInfo.extensions << std::endl;
    return outputTarget;
}

} // namespace ktt
