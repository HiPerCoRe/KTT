#pragma once

#ifdef KTT_API_OPENCL

#include <CL/cl.h>

#include <ComputeEngine/OpenCl/OpenClDevice.h>
#include <ComputeEngine/OpenCl/OpenClPlatform.h>
#include <KttTypes.h>

namespace ktt
{

class OpenClContext
{
public:
    explicit OpenClContext(const OpenClPlatform& platform, const OpenClDevice& device);
    explicit OpenClContext(ComputeContext context);
    ~OpenClContext();

    cl_context GetContext() const;
    cl_device_id GetDevice() const;

private:
    cl_context m_Context;
    cl_device_id m_Device;
    bool m_OwningContext;
};

} // namespace ktt

#endif // KTT_API_OPENCL