#pragma once

#ifdef KTT_PLATFORM_OPENCL

#include <CL/cl.h>
#include <ktt_types.h>

namespace ktt
{

class OpenCLContext
{
public:
    explicit OpenCLContext(const cl_platform_id platform, const cl_device_id device);
    explicit OpenCLContext(UserContext context);
    ~OpenCLContext();

    cl_context getContext() const;
    cl_platform_id getPlatform() const;
    cl_device_id getDevice() const;

private:
    cl_context context;
    cl_platform_id platform;
    cl_device_id device;
    bool owningContext;
};

} // namespace ktt

#endif // KTT_PLATFORM_OPENCL
