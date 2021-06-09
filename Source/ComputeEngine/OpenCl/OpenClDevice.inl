#ifdef KTT_API_OPENCL

#include <CL/cl.h>

#include <ComputeEngine/OpenCl/OpenClDevice.h>
#include <ComputeEngine/OpenCl/OpenClUtility.h>

namespace ktt
{

template <typename T>
T OpenClDevice::GetInfoWithType(const cl_device_info info) const
{
    T result;
    CheckError(clGetDeviceInfo(m_Id, info, sizeof(result), &result, nullptr), "clGetDeviceInfo");
    return result;
}

} // namespace ktt

#endif // KTT_API_OPENCL
