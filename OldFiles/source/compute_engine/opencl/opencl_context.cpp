#ifdef KTT_PLATFORM_OPENCL

#include <stdexcept>
#include <vector>
#include <compute_engine/opencl/opencl_context.h>
#include <compute_engine/opencl/opencl_utility.h>

namespace ktt
{

OpenCLContext::OpenCLContext(const cl_platform_id platform, const cl_device_id device) :
    platform(platform),
    device(device),
    owningContext(true)
{
    cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
    cl_int result;
    context = clCreateContext(properties, 1, &device, nullptr, nullptr, &result);
    checkOpenCLError(result, "clCreateContext");
}

OpenCLContext::OpenCLContext(UserContext context) :
    owningContext(false)
{
    this->context = static_cast<cl_context>(context);

    if (this->context == nullptr)
    {
        throw std::runtime_error("The provided user OpenCL context is not valid");
    }

    cl_uint deviceCount;
    checkOpenCLError(clGetContextInfo(this->context, CL_CONTEXT_NUM_DEVICES, sizeof(cl_uint), &deviceCount, nullptr), "clGetContextInfo");

    if (deviceCount != 1)
    {
        throw std::runtime_error("The provided user OpenCL context must have device count equal to 1");
    }

    cl_device_id device;
    checkOpenCLError(clGetContextInfo(this->context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &device, nullptr), "clGetContextInfo");
    this->device = device;

    cl_platform_id platform;
    checkOpenCLError(clGetDeviceInfo(this->device, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &platform, nullptr), "clGetDeviceInfo");
    this->platform = platform;
}

OpenCLContext::~OpenCLContext()
{
    if (owningContext)
    {
        checkOpenCLError(clReleaseContext(context), "clReleaseContext");
    }
}

cl_context OpenCLContext::getContext() const
{
    return context;
}

cl_platform_id OpenCLContext::getPlatform() const
{
    return platform;
}

cl_device_id OpenCLContext::getDevice() const
{
    return device;
}

} // namespace ktt

#endif // KTT_PLATFORM_OPENCL
