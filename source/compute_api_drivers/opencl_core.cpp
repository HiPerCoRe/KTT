#include "opencl_core.h"

#include "CL/cl.h"

namespace ktt
{

OpenCLCore::OpenCLCore() = default;

std::vector<OpenCLPlatform> OpenCLCore::getOpenCLPlatforms() const
{
    cl_uint platformCount;
    clGetPlatformIDs(0, nullptr, &platformCount);

    std::vector<cl_platform_id> platformIds(platformCount);
    clGetPlatformIDs(platformCount, platformIds.data(), nullptr);

    std::vector<OpenCLPlatform> platforms;
    for (const auto platformId : platformIds)
    {
        std::string name = getPlatformInfo(platformId, CL_PLATFORM_NAME);
        std::string version = getPlatformInfo(platformId, CL_PLATFORM_VERSION);
        std::string vendor = getPlatformInfo(platformId, CL_PLATFORM_VENDOR);

        platforms.push_back(OpenCLPlatform(platformId, version, name, vendor));
    }

    return platforms;
}

std::vector<OpenCLDevice> OpenCLCore::getOpenCLDevices(const OpenCLPlatform& platform) const
{
    cl_uint deviceCount;
    clGetDeviceIDs(platform.getId(), CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount);

    throw std::runtime_error("Unfinished method");
    // to do

    return std::vector<OpenCLDevice>();
}

std::string OpenCLCore::getPlatformInfo(const cl_platform_id id, const cl_platform_info info) const
{
    size_t infoSize;
    clGetPlatformInfo(id, info, 0, nullptr, &infoSize);
    std::string infoString(infoSize, ' ');
    clGetPlatformInfo(id, info, infoSize, &infoString[0], nullptr);
    
    return infoString;
}

} // namespace ktt
