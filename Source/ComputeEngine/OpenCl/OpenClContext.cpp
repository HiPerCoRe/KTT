#ifdef KTT_API_OPENCL

#include <vector>

#include <ComputeEngine/OpenCl/OpenClContext.h>
#include <ComputeEngine/OpenCl/OpenClUtility.h>
#include <Utility/ErrorHandling/KttException.h>

namespace ktt
{

OpenClContext::OpenClContext(const OpenClPlatform& platform, const OpenClDevice& device) :
    m_Device(device.GetId()),
    m_OwningContext(true)
{
    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform.GetId()), 0};
    cl_int result;
    m_Context = clCreateContext(properties, 1, &m_Device, nullptr, nullptr, &result);
    CheckError(result, "clCreateContext");
}

OpenClContext::OpenClContext(ComputeContext context) :
    m_OwningContext(false)
{
    m_Context = static_cast<cl_context>(context);

    if (m_Context == nullptr)
    {
        throw KttException("The provided user OpenCL context is not valid");
    }

    cl_uint deviceCount;
    CheckError(clGetContextInfo(m_Context, CL_CONTEXT_NUM_DEVICES, sizeof(deviceCount), &deviceCount, nullptr),
        "clGetContextInfo");

    if (deviceCount != 1)
    {
        throw KttException("The provided user OpenCL context must have number of devices equal to 1");
    }

    cl_device_id device;
    CheckError(clGetContextInfo(m_Context, CL_CONTEXT_DEVICES, sizeof(device), &device, nullptr), "clGetContextInfo");
    m_Device = device;
}

OpenClContext::~OpenClContext()
{
    if (m_OwningContext)
    {
        CheckError(clReleaseContext(m_Context), "clReleaseContext");
    }
}

cl_context OpenClContext::GetContext() const
{
    return m_Context;
}

cl_device_id OpenClContext::GetDevice() const
{
    return m_Device;
}

} // namespace ktt

#endif // KTT_PLATFORM_OPENCL