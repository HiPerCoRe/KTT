#ifdef KTT_API_OPENCL

#include <ComputeEngine/OpenCl/OpenClPlatform.h>
#include <ComputeEngine/OpenCl/OpenClUtility.h>

namespace ktt
{

OpenClPlatform::OpenClPlatform(const PlatformIndex index, const cl_platform_id id) :
    m_Index(index),
    m_Id(id)
{}

PlatformIndex OpenClPlatform::GetIndex() const
{
    return m_Index;
}

cl_platform_id OpenClPlatform::GetId() const
{
    return m_Id;
}

PlatformInfo OpenClPlatform::GetInfo() const
{
    const std::string name = GetInfoString(CL_PLATFORM_NAME);
    const std::string version = GetInfoString(CL_PLATFORM_VERSION);
    const std::string vendor = GetInfoString(CL_PLATFORM_VENDOR);
    const std::string extensions = GetInfoString(CL_PLATFORM_EXTENSIONS);

    PlatformInfo result(m_Index, name);
    result.SetVersion(version);
    result.SetVendor(vendor);
    result.SetExtensions(extensions);

    return result;
}

std::vector<OpenClDevice> OpenClPlatform::GetDevices() const
{
    cl_uint deviceCount;
    CheckError(clGetDeviceIDs(m_Id, CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount), "clGetDeviceIDs");

    std::vector<cl_device_id> deviceIds(deviceCount);
    CheckError(clGetDeviceIDs(m_Id, CL_DEVICE_TYPE_ALL, deviceCount, deviceIds.data(), nullptr), "clGetDeviceIDs");

    std::vector<OpenClDevice> devices;

    for (DeviceIndex index = 0; index < static_cast<DeviceIndex>(deviceIds.size()); ++index)
    {
        devices.emplace_back(index, deviceIds[static_cast<size_t>(index)]);
    }

    return devices;
}

std::vector<OpenClPlatform> OpenClPlatform::GetAllPlatforms()
{
    cl_uint platformCount;
    CheckError(clGetPlatformIDs(0, nullptr, &platformCount), "clGetPlatformIDs");

    std::vector<cl_platform_id> platformIds(platformCount);
    CheckError(clGetPlatformIDs(platformCount, platformIds.data(), nullptr), "clGetPlatformIDs");

    std::vector<OpenClPlatform> platforms;

    for (PlatformIndex index = 0; index < static_cast<PlatformIndex>(platformIds.size()); ++index)
    {
        platforms.emplace_back(index, platformIds[static_cast<size_t>(index)]);
    }

    return platforms;
}

std::string OpenClPlatform::GetInfoString(const cl_platform_info info) const
{
    size_t infoSize;
    CheckError(clGetPlatformInfo(m_Id, info, 0, nullptr, &infoSize), "clGetPlatformInfo");

    std::string infoString(infoSize, ' ');
    CheckError(clGetPlatformInfo(m_Id, info, infoSize, infoString.data(), nullptr), "clGetPlatformInfo");

    return infoString;
}

} // namespace ktt

#endif // KTT_API_OPENCL
