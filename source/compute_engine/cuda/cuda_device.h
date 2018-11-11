#pragma once

#include <string>
#include <cuda.h>

namespace ktt
{

class CUDADevice
{
public:
    explicit CUDADevice(const CUdevice device, const std::string& name) :
        device(device),
        name(name)
    {}

    CUdevice getDevice() const
    {
        return device;
    }

    const std::string& getName() const
    {
        return name;
    }

private:
    CUdevice device;
    std::string name;
};

} // namespace ktt
