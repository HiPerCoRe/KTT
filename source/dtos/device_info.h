#pragma once

#include <cstdint>
#include <string>

#include "../enums/device_type.h"

namespace ktt
{

class DeviceInfo
{
public:
    explicit DeviceInfo(const size_t id, const std::string& name):
        id(id),
        name(name)
    {}

    size_t getId() const
    {
        return id;
    }

    std::string getName() const
    {
        return name;
    }

    std::string getVendor() const
    {
        return vendor;
    }

    std::string getExtensions() const
    {
        return extensions;
    }

    DeviceType getDeviceType() const
    {
        return deviceType;
    }

    uint64_t getGlobalMemorySize() const
    {
        return globalMemorySize;
    }

    uint64_t getLocalMemorySize() const
    {
        return localMemorySize;
    }

    uint64_t getMaxConstantBufferSize() const
    {
        return maxConstantBufferSize;
    }

    uint32_t getMaxComputeUnits() const
    {
        return maxComputeUnits;
    }

    size_t getMaxWorkGroupSize() const
    {
        return maxWorkGroupSize;
    }

    void setVendor(const std::string& vendor)
    {
        this->vendor = vendor;
    }

    void setExtensions(const std::string& extensions)
    {
        this->extensions = extensions;
    }

    void setDeviceType(const DeviceType& deviceType)
    {
        this->deviceType = deviceType;
    }

    void setGlobalMemorySize(const uint64_t globalMemorySize)
    {
        this->globalMemorySize = globalMemorySize;
    }

    void setLocalMemorySize(const uint64_t localMemorySize)
    {
        this->localMemorySize = localMemorySize;
    }

    void setMaxConstantBufferSize(const uint64_t maxConstantBufferSize)
    {
        this->maxConstantBufferSize = maxConstantBufferSize;
    }

    void setMaxComputeUnits(const uint32_t maxComputeUnits)
    {
        this->maxComputeUnits = maxComputeUnits;
    }

    void setMaxWorkGroupSize(const size_t maxWorkGroupSize)
    {
        this->maxWorkGroupSize = maxWorkGroupSize;
    }

private:
    size_t id;
    std::string name;
    std::string vendor;
    std::string extensions;
    DeviceType deviceType;
    uint64_t globalMemorySize;
    uint64_t localMemorySize;
    uint64_t maxConstantBufferSize;
    uint32_t maxComputeUnits;
    size_t maxWorkGroupSize;
};

} // namespace ktt
