#pragma once

#include <string>

#include "cuda.h"

namespace ktt
{

class CudaDevice
{
public:
    explicit CudaDevice(const CUdevice device, const std::string& name) :
        device(device),
        name(name)
    {}

    CUdevice getDevice() const
    {
        return device;
    }

    std::string getName() const
    {
        return name;
    }

private:
    CUdevice device;
    std::string name;
};

} // namespace ktt
