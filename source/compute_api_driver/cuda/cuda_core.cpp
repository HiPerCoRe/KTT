#include "cuda_core.h"

namespace ktt
{

#ifdef PLATFORM_CUDA

CudaCore::CudaCore(const size_t deviceIndex) :
    deviceIndex(deviceIndex),
    compilerOptions(std::string("")),
    useReadBufferCache(true),
    useWriteBufferCache(false),
    useReadWriteBufferCache(false)
{
    checkCudaError(cuInit(0), "cuInit");

    auto devices = getCudaDevices();
    if (deviceIndex >= devices.size())
    {
        throw std::runtime_error(std::string("Invalid device index: ") + std::to_string(deviceIndex));
    }
    context = std::make_unique<CudaContext>(devices.at(deviceIndex).getDevice());
}

void CudaCore::printComputeApiInfo(std::ostream& outputTarget) const
{
    outputTarget << "Platform 0: " << "NVIDIA CUDA" << std::endl;
    auto devices = getCudaDevices();

    for (size_t i = 0; i < devices.size(); i++)
    {
        outputTarget << "Device " << i << ": " << devices.at(i).getName() << std::endl;
    }
    outputTarget << std::endl;
}

std::vector<PlatformInfo> CudaCore::getPlatformInfo() const
{
    int driverVersion;
    checkCudaError(cuDriverGetVersion(&driverVersion), "cuDriverGetVersion");

    PlatformInfo cuda(0, "NVIDIA CUDA");
    cuda.setVendor("NVIDIA Corporation");
    cuda.setVersion(std::to_string(driverVersion));
    cuda.setExtensions("");
    return std::vector<PlatformInfo>{ cuda };
}

std::vector<DeviceInfo> CudaCore::getDeviceInfo(const size_t) const
{
    std::vector<DeviceInfo> result;
    auto devices = getCudaDevices();

    for (size_t i = 0; i < devices.size(); i++)
    {
        result.push_back(getCudaDeviceInfo(i));
    }

    return result;
}

DeviceInfo CudaCore::getCurrentDeviceInfo() const
{
    return getCudaDeviceInfo(deviceIndex);
}

void CudaCore::setCompilerOptions(const std::string& options)
{
    compilerOptions = options;
}

void CudaCore::setCacheUsage(const bool flag, const ArgumentMemoryType& argumentMemoryType)
{
    switch (argumentMemoryType)
    {
    case ArgumentMemoryType::ReadOnly:
        useReadBufferCache = flag;
        break;
    case ArgumentMemoryType::WriteOnly:
        useWriteBufferCache = flag;
        break;
    case ArgumentMemoryType::ReadWrite:
        useReadWriteBufferCache = flag;
        break;
    default:
        throw std::runtime_error("Unknown argument memory type");
    }
}

void CudaCore::clearCache()
{
    throw std::runtime_error("clearCache() method is not supported yet for CUDA platform");
}

void CudaCore::clearCache(const ArgumentMemoryType& argumentMemoryType)
{
    throw std::runtime_error("clearCache() method is not supported yet for CUDA platform");
}

KernelRunResult CudaCore::runKernel(const std::string& source, const std::string& kernelName, const std::vector<size_t>& globalSize,
    const std::vector<size_t>& localSize, const std::vector<const KernelArgument*>& argumentPointers)
{
    throw std::runtime_error("runKernel() method is not supported yet for CUDA platform");
}

std::vector<CudaDevice> CudaCore::getCudaDevices() const
{
    int deviceCount;
    checkCudaError(cuDeviceGetCount(&deviceCount), "cuDeviceGetCount");

    std::vector<CUdevice> deviceIds(deviceCount);
    for (int i = 0; i < deviceCount; i++)
    {
        checkCudaError(cuDeviceGet(&deviceIds.at(i), i), "cuDeviceGet");
    }

    std::vector<CudaDevice> devices;
    for (const auto deviceId : deviceIds)
    {
        std::string name(50, ' ');
        checkCudaError(cuDeviceGetName(&name[0], 50, deviceId), "cuDeviceGetName");
        devices.push_back(CudaDevice(deviceId, name));
    }

    return devices;
}

DeviceInfo CudaCore::getCudaDeviceInfo(const size_t deviceIndex) const
{
    auto devices = getCudaDevices();
    DeviceInfo result(deviceIndex, devices.at(deviceIndex).getName());

    CUdevice id = devices.at(deviceIndex).getDevice();
    result.setExtensions("");
    result.setVendor("NVIDIA Corporation");
    
    size_t globalMemory;
    checkCudaError(cuDeviceTotalMem(&globalMemory, id), "cuDeviceTotalMem");
    result.setGlobalMemorySize(globalMemory);

    int localMemory;
    checkCudaError(cuDeviceGetAttribute(&localMemory, CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK, id), "cuDeviceGetAttribute");
    result.setLocalMemorySize(localMemory);

    int constantMemory;
    checkCudaError(cuDeviceGetAttribute(&constantMemory, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, id), "cuDeviceGetAttribute");
    result.setMaxConstantBufferSize(constantMemory);

    int computeUnits;
    checkCudaError(cuDeviceGetAttribute(&computeUnits, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, id), "cuDeviceGetAttribute");
    result.setMaxComputeUnits(computeUnits);

    int workGroupSize;
    checkCudaError(cuDeviceGetAttribute(&workGroupSize, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, id), "cuDeviceGetAttribute");
    result.setMaxWorkGroupSize(workGroupSize);
    result.setDeviceType(DeviceType::GPU);

    return result;
}

#else

CudaCore::CudaCore(const size_t)
{
    throw std::runtime_error("Current platform does not support CUDA or CUDA build option was not specified during project file generation");
}

void CudaCore::printComputeApiInfo(std::ostream&) const
{
    throw std::runtime_error("Current platform does not support CUDA or CUDA build option was not specified during project file generation");
}

std::vector<PlatformInfo> CudaCore::getPlatformInfo() const
{
    throw std::runtime_error("Current platform does not support CUDA or CUDA build option was not specified during project file generation");
}

std::vector<DeviceInfo> CudaCore::getDeviceInfo(const size_t) const
{
    throw std::runtime_error("Current platform does not support CUDA or CUDA build option was not specified during project file generation");
}

DeviceInfo CudaCore::getCurrentDeviceInfo() const
{
    throw std::runtime_error("Current platform does not support CUDA or CUDA build option was not specified during project file generation");
}

void CudaCore::setCompilerOptions(const std::string&)
{
    throw std::runtime_error("Current platform does not support CUDA or CUDA build option was not specified during project file generation");
}

void CudaCore::setCacheUsage(const bool, const ArgumentMemoryType&)
{
    throw std::runtime_error("Current platform does not support CUDA or CUDA build option was not specified during project file generation");
}

void CudaCore::clearCache()
{
    throw std::runtime_error("Current platform does not support CUDA or CUDA build option was not specified during project file generation");
}

void CudaCore::clearCache(const ArgumentMemoryType&)
{
    throw std::runtime_error("Current platform does not support CUDA or CUDA build option was not specified during project file generation");
}

KernelRunResult CudaCore::runKernel(const std::string&, const std::string&, const std::vector<size_t>&, const std::vector<size_t>&,
    const std::vector<const KernelArgument*>&)
{
    throw std::runtime_error("Current platform does not support CUDA or CUDA build option was not specified during project file generation");
}

#endif // PLATFORM_CUDA

} // namespace ktt
