#ifdef KTT_API_OPENCL

#include <Api/KttException.h>
#include <ComputeEngine/OpenCl/OpenClContext.h>
#include <ComputeEngine/OpenCl/OpenClDevice.h>
#include <ComputeEngine/OpenCl/OpenClPlatform.h>
#include <ComputeEngine/OpenCl/OpenClUtility.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

OpenClContext::OpenClContext(const OpenClPlatform& platform, const OpenClDevice& device) :
    m_Platform(platform.GetId()),
    m_Device(device.GetId()),
    m_OwningContext(true)
{
    Logger::LogDebug("Initializing OpenCL context");
    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform.GetId()), 0};
    cl_int result;
    m_Context = clCreateContext(properties, 1, &m_Device, nullptr, nullptr, &result);
    CheckError(result, "clCreateContext");
}

OpenClContext::OpenClContext(ComputeContext context) :
    m_OwningContext(false)
{
    Logger::LogDebug("Initializing OpenCL context");
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

    cl_platform_id platform;
    CheckError(clGetDeviceInfo(m_Device, CL_DEVICE_PLATFORM, sizeof(platform), &platform, nullptr), "clGetDeviceInfo");
    m_Platform = platform;
}

OpenClContext::~OpenClContext()
{
    Logger::LogDebug("Releasing OpenCL context");

    if (m_OwningContext)
    {
        CheckError(clReleaseContext(m_Context), "clReleaseContext");
    }
}

cl_context OpenClContext::GetContext() const
{
    return m_Context;
}

cl_platform_id OpenClContext::GetPlatform() const
{
    return m_Platform;
}

cl_device_id OpenClContext::GetDevice() const
{
    return m_Device;
}

} // namespace ktt

#endif // KTT_PLATFORM_OPENCL
