#ifdef KTT_API_CUDA

#include <cstddef>

#include <ComputeEngine/Cuda/CudaDevice.h>
#include <ComputeEngine/Cuda/CudaUtility.h>

namespace ktt
{

CudaDevice::CudaDevice(const DeviceIndex index, const CUdevice device) :
    m_Index(index),
    m_Device(device)
{}

DeviceIndex CudaDevice::GetIndex() const
{
    return m_Index;
}

CUdevice CudaDevice::GetDevice() const
{
    return m_Device;
}

DeviceInfo CudaDevice::GetInfo() const
{
    char name[100];
    CheckError(cuDeviceGetName(name, 100, m_Device), "cuDeviceGetName");

    DeviceInfo result(m_Index, name);
    result.SetVendor("NVIDIA Corporation");
    result.SetExtensions("N/A");
    result.SetDeviceType(DeviceType::GPU);

    size_t globalMemory;
    CheckError(cuDeviceTotalMem(&globalMemory, m_Device), "cuDeviceTotalMem");
    result.SetGlobalMemorySize(globalMemory);

    const int localMemory = GetAttribute(CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK);
    result.SetLocalMemorySize(static_cast<uint64_t>(localMemory));

    const int constantMemory = GetAttribute(CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY);
    result.SetMaxConstantBufferSize(static_cast<uint64_t>(constantMemory));

    const int workGroupSize = GetAttribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
    result.SetMaxWorkGroupSize(static_cast<uint64_t>(workGroupSize));

    const int computeUnits = GetAttribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
    result.SetMaxComputeUnits(static_cast<uint32_t>(computeUnits));

    const int computeCapabilityMajor = GetAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
    result.SetCudaComputeCapabilityMajor(static_cast<uint32_t>(computeCapabilityMajor));

    const int computeCapabilityMinor = GetAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
    result.SetCudaComputeCapabilityMinor(static_cast<uint32_t>(computeCapabilityMinor));

    return result;
}

std::vector<CudaDevice> CudaDevice::GetAllDevices()
{
    int deviceCount;
    CheckError(cuDeviceGetCount(&deviceCount), "cuDeviceGetCount");
    std::vector<CUdevice> deviceIds(static_cast<size_t>(deviceCount));

    for (size_t i = 0; i < static_cast<size_t>(deviceCount); ++i)
    {
        CheckError(cuDeviceGet(&deviceIds[i], static_cast<int>(i)), "cuDeviceGet");
    }

    std::vector<CudaDevice> devices;

    for (size_t i = 0; i < deviceIds.size(); ++i)
    {
        devices.emplace_back(static_cast<DeviceIndex>(i), deviceIds[i]);
    }

    return devices;
}

int CudaDevice::GetAttribute(const CUdevice_attribute attribute) const
{
    int result;
    CheckError(cuDeviceGetAttribute(&result, attribute, m_Device), "cuDeviceGetAttribute");
    return result;
}

} // namespace ktt

#endif // KTT_API_CUDA
