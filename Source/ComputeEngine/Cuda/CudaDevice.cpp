#ifdef KTT_API_CUDA

#include <cstddef>
#include <cstdio>
#include <string>
#include <nvml.h>

#include <ComputeEngine/Cuda/CudaDevice.h>
#include <ComputeEngine/Cuda/CudaUtility.h>

namespace ktt
{

// NVIDIA driver version (e.g. "550.54.14"). The CUDA driver API only reports the supported CUDA version, so the
// actual driver version is taken from NVML. It is the same for all devices, so it is queried once and cached.
// Left empty if NVML cannot be initialized.
static const std::string& GetNvidiaDriverVersion()
{
    static const std::string version = []()
    {
        if (nvmlInit_v2() != NVML_SUCCESS)
        {
            return std::string();
        }

        char buffer[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];
        const bool success = nvmlSystemGetDriverVersion(buffer, sizeof(buffer)) == NVML_SUCCESS;
        nvmlShutdown();
        return success ? std::string(buffer) : std::string();
    }();

    return version;
}

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
    result.SetDriverVersion(GetNvidiaDriverVersion());

    // Persistent hardware identifier: the device UUID, formatted the same way as nvidia-smi ("GPU-<uuid>").
    // cuDeviceGetUuid is available since CUDA 9.2; leave the identifier empty if the driver rejects the call.
    CUuuid uuid;
    if (cuDeviceGetUuid(&uuid, m_Device) == CUDA_SUCCESS)
    {
        const auto* bytes = reinterpret_cast<const unsigned char*>(uuid.bytes);
        char identifier[45];
        std::snprintf(identifier, sizeof(identifier),
            "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]);
        result.SetDeviceIdentifier(identifier);
    }

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
