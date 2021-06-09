#pragma once

#ifdef KTT_API_OPENCL

#include <string>
#include <vector>
#include <CL/cl.h>

#include <Api/Info/PlatformInfo.h>
#include <ComputeEngine/OpenCl/OpenClDevice.h>
#include <KttTypes.h>

namespace ktt
{

class OpenClPlatform
{
public:
    explicit OpenClPlatform(const PlatformIndex index, const cl_platform_id id);

    PlatformIndex GetIndex() const;
    cl_platform_id GetId() const;
    PlatformInfo GetInfo() const;
    std::vector<OpenClDevice> GetDevices() const;

    static std::vector<OpenClPlatform> GetAllPlatforms();

private:
    PlatformIndex m_Index;
    cl_platform_id m_Id;

    std::string GetInfoString(const cl_platform_info info) const;
};

} // namespace ktt

#endif // KTT_API_OPENCL
