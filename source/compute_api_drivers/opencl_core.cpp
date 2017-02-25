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

    std::vector<cl_device_id> deviceIds(deviceCount);
    clGetDeviceIDs(platform.getId(), CL_DEVICE_TYPE_ALL, deviceCount, deviceIds.data(), nullptr);

    std::vector<OpenCLDevice> devices;
    for (const auto deviceId : deviceIds)
    {
        std::string name = getDeviceInfo(deviceId, CL_DEVICE_NAME);
        std::string vendor = getDeviceInfo(deviceId, CL_DEVICE_VENDOR);

        devices.push_back(OpenCLDevice(deviceId, name, vendor));
    }

    return devices;
}

void OpenCLCore::printOpenCLInfo(std::ostream& outputTarget) const
{
    std::vector<ktt::OpenCLPlatform> platforms = getOpenCLPlatforms();

    for (size_t i = 0; i < platforms.size(); i++)
    {
        outputTarget << "Platform " << i << ": " << platforms.at(i).getVendor() << " " << platforms.at(i).getName() << std::endl;
        std::vector<ktt::OpenCLDevice> devices = getOpenCLDevices(platforms.at(i));

        outputTarget << "Devices for platform " << i << ":" << std::endl;
        for (size_t j = 0; j < devices.size(); j++)
        {
            outputTarget << "Device " << j << ": " << devices.at(j).getVendor() << " " << devices.at(j).getName() << std::endl;
        }
        outputTarget << std::endl;
    }
}

std::string OpenCLCore::getPlatformInfo(const cl_platform_id id, const cl_platform_info info) const
{
    size_t infoSize;
    clGetPlatformInfo(id, info, 0, nullptr, &infoSize);
    std::string infoString(infoSize, ' ');
    clGetPlatformInfo(id, info, infoSize, &infoString[0], nullptr);
    
    return infoString;
}

std::string OpenCLCore::getDeviceInfo(const cl_device_id id, const cl_device_info info) const
{
    size_t infoSize;
    clGetDeviceInfo(id, info, 0, nullptr, &infoSize);
    std::string infoString(infoSize, ' ');
    clGetDeviceInfo(id, info, infoSize, &infoString[0], nullptr);

    return infoString;
}

} // namespace ktt
