#include "opencl_core.h"

#include "CL/cl.h"

namespace ktt
{

OpenCLCore::OpenCLCore() = default;

std::vector<OpenCLPlatform> OpenCLCore::getOpenCLPlatforms() const
{
    cl_uint platformsCount;
    clGetPlatformIDs(0, nullptr, &platformsCount);

    std::vector<cl_platform_id> platformIds(platformsCount);
    clGetPlatformIDs(platformsCount, platformIds.data(), nullptr);

    std::vector<OpenCLPlatform> platforms;
    for (const auto platformId : platformIds)
    {
        size_t nameSize;
        clGetPlatformInfo(platformId, CL_PLATFORM_NAME, 0, nullptr, &nameSize);
        std::string name(nameSize, ' ');
        clGetPlatformInfo(platformId, CL_PLATFORM_NAME, nameSize, (void*)name.data(), nullptr);

        size_t versionSize;
        clGetPlatformInfo(platformId, CL_PLATFORM_VERSION, 0, nullptr, &versionSize);
        std::string version(versionSize, ' ');
        clGetPlatformInfo(platformId, CL_PLATFORM_VERSION, versionSize, (void*)version.data(), nullptr);

        size_t vendorSize;
        clGetPlatformInfo(platformId, CL_PLATFORM_VENDOR, 0, nullptr, &vendorSize);
        std::string vendor(vendorSize, ' ');
        clGetPlatformInfo(platformId, CL_PLATFORM_VENDOR, vendorSize, (void*)vendor.data(), nullptr);

        platforms.push_back(OpenCLPlatform(platformId, version, name, vendor));
    }

    return platforms;
}

std::vector<OpenCLDevice> OpenCLCore::getOpenCLDevices(const OpenCLPlatform& platform) const
{
    cl_uint devicesCount;
    clGetDeviceIDs(platform.getId(), CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount);

    throw std::runtime_error("Unfinished method");
    // to do

    return std::vector<OpenCLDevice>();
}

} // namespace ktt
