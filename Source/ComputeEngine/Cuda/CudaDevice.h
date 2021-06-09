#pragma once

#ifdef KTT_API_CUDA

#include <string>
#include <vector>
#include <cuda.h>

#include <Api/Info/DeviceInfo.h>
#include <KttTypes.h>

namespace ktt
{

class CudaDevice
{
public:
    explicit CudaDevice(const DeviceIndex index, const CUdevice device);

    DeviceIndex GetIndex() const;
    CUdevice GetDevice() const;
    DeviceInfo GetInfo() const;

    static std::vector<CudaDevice> GetAllDevices();

private:
    DeviceIndex m_Index;
    CUdevice m_Device;

    int GetAttribute(const CUdevice_attribute attribute) const;
};

} // namespace ktt

#endif // KTT_API_CUDA
