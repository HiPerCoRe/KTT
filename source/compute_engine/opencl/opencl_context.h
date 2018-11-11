#pragma once

#include <vector>
#include <CL/cl.h>
#include <compute_engine/opencl/opencl_utility.h>

namespace ktt
{

class OpenCLContext
{
public:
    explicit OpenCLContext(const cl_platform_id platform, const std::vector<cl_device_id>& devices) :
        platform(platform),
        devices(devices)
    {
        cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
        cl_int result;
        context = clCreateContext(properties, static_cast<cl_uint>(devices.size()), devices.data(), nullptr, nullptr, &result);
        checkOpenCLError(result, "clCreateContext");
    }

    ~OpenCLContext()
    {
        checkOpenCLError(clReleaseContext(context), "clReleaseContext");
    }

    cl_platform_id getPlatform() const
    {
        return platform;
    }

    const std::vector<cl_device_id>& getDevices() const
    {
        return devices;
    }

    cl_context getContext() const
    {
        return context;
    }

private:
    cl_platform_id platform;
    std::vector<cl_device_id> devices;
    cl_context context;
};

} // namespace ktt
