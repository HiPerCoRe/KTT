#include "cuda_core.h"

namespace ktt
{

#ifdef PLATFORM_CUDA

CudaCore::CudaCore(const size_t deviceIndex) :
    deviceIndex(deviceIndex),
    compilerOptions(std::string(""))
{
    checkCudaError(cuInit(0), "cuInit");
}

void CudaCore::printComputeApiInfo(std::ostream& outputTarget) const
{
    auto devices = getCudaDevices();

    for (size_t i = 0; i < devices.size(); i++)
    {
        outputTarget << "Device " << i << ": " << devices.at(i).getName() << std::endl;
    }
    outputTarget << std::endl;
}

std::vector<PlatformInfo> CudaCore::getPlatformInfo() const
{
    return std::vector<PlatformInfo>{};
}

std::vector<DeviceInfo> CudaCore::getDeviceInfo(const size_t platformIndex) const
{
    throw std::runtime_error("getDeviceInfo() method is not supported yet for CUDA platform");
}

void CudaCore::setCompilerOptions(const std::string& options)
{
    compilerOptions = options;
}

void CudaCore::setCacheUsage(const bool flag, const ArgumentMemoryType& argumentMemoryType)
{
    throw std::runtime_error("setCacheUsage() method is not supported yet for CUDA platform");
}

void CudaCore::clearCache()
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
        std::string name(100, ' ');
        checkCudaError(cuDeviceGetName(&name[0], 100, deviceId), "cuDeviceGetName");
        devices.push_back(CudaDevice(deviceId, name));
    }

    return devices;
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

KernelRunResult CudaCore::runKernel(const std::string&, const std::string&, const std::vector<size_t>&, const std::vector<size_t>&,
    const std::vector<const KernelArgument*>&)
{
    throw std::runtime_error("Current platform does not support CUDA or CUDA build option was not specified during project file generation");
}

#endif // PLATFORM_CUDA

} // namespace ktt
