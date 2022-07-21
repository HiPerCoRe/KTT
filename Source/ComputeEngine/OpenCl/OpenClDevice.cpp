#ifdef KTT_API_OPENCL

#include <ComputeEngine/OpenCl/OpenClDevice.h>
#include <ComputeEngine/OpenCl/OpenClUtility.h>
#include <Utility/StringUtility.h>

namespace ktt
{

OpenClDevice::OpenClDevice(const DeviceIndex index, const cl_device_id id) :
    m_Index(index),
    m_Id(id)
{}

DeviceIndex OpenClDevice::GetIndex() const
{
    return m_Index;
}

cl_device_id OpenClDevice::GetId() const
{
    return m_Id;
}

DeviceType OpenClDevice::GetDeviceType() const
{
    const auto deviceType = GetInfoWithType<cl_device_type>(CL_DEVICE_TYPE);

    switch (deviceType)
    {
    case CL_DEVICE_TYPE_CPU:
        return DeviceType::CPU;
    case CL_DEVICE_TYPE_GPU:
        return DeviceType::GPU;
    default:
        return DeviceType::Custom;
    }
}

DeviceInfo OpenClDevice::GetInfo() const
{
    const std::string name = GetInfoString(CL_DEVICE_NAME);
    const std::string vendor = GetInfoString(CL_DEVICE_VENDOR);
    const std::string extensions = GetInfoString(CL_DEVICE_EXTENSIONS);

    DeviceInfo result(m_Index, name);
    result.SetVendor(vendor);
    result.SetExtensions(extensions);

    const DeviceType type = GetDeviceType();
    result.SetDeviceType(type);

    const auto globalMemorySize = GetInfoWithType<uint64_t>(CL_DEVICE_GLOBAL_MEM_SIZE);
    result.SetGlobalMemorySize(globalMemorySize);

    const auto localMemorySize = GetInfoWithType<uint64_t>(CL_DEVICE_LOCAL_MEM_SIZE);
    result.SetLocalMemorySize(localMemorySize);

    const auto maxConstantBufferSize = GetInfoWithType<uint64_t>(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE);
    result.SetMaxConstantBufferSize(maxConstantBufferSize);

    const auto maxWorkGroupSize = GetInfoWithType<size_t>(CL_DEVICE_MAX_WORK_GROUP_SIZE);
    result.SetMaxWorkGroupSize(static_cast<uint64_t>(maxWorkGroupSize));

    const auto maxComputeUnits = GetInfoWithType<uint32_t>(CL_DEVICE_MAX_COMPUTE_UNITS);
    result.SetMaxComputeUnits(maxComputeUnits);

    result.SetCudaComputeCapabilityMajor(0);
    result.SetCudaComputeCapabilityMinor(0);

    return result;
}

std::string OpenClDevice::GetInfoString(const cl_device_info info) const
{
    size_t infoSize;
    CheckError(clGetDeviceInfo(m_Id, info, 0, nullptr, &infoSize), "clGetDeviceInfo");

    std::string infoString(infoSize, '\0');
    CheckError(clGetDeviceInfo(m_Id, info, infoSize, infoString.data(), nullptr), "clGetDeviceInfo");

    RemoveTrailingZero(infoString);
    return infoString;
}

} // namespace ktt

#endif // KTT_API_OPENCL
