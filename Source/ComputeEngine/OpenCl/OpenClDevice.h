#pragma once

#ifdef KTT_API_OPENCL

#include <string>
#include <CL/cl.h>

#include <Api/Info/DeviceInfo.h>
#include <KttTypes.h>

namespace ktt
{

class OpenClDevice
{
public:
    explicit OpenClDevice(const DeviceIndex index, const cl_device_id id);

    DeviceIndex GetIndex() const;
    cl_device_id GetId() const;
    DeviceType GetDeviceType() const;
    DeviceInfo GetInfo() const;

private:
    DeviceIndex m_Index;
    cl_device_id m_Id;

    std::string GetInfoString(const cl_device_info info) const;

    template <typename T>
    T GetInfoWithType(const cl_device_info info) const;
};

} // namespace ktt

#include <ComputeEngine/OpenCl/OpenClDevice.inl>

#endif // KTT_API_OPENCL
