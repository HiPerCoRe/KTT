#include <stdexcept>

#include "cuda_core.h"

namespace ktt
{

#ifdef USE_CUDA

// to do

#else

CudaCore::CudaCore(const size_t deviceIndex) :
    deviceIndex(deviceIndex)
{}

void CudaCore::printComputeApiInfo(std::ostream& outputTarget) const
{
    throw std::runtime_error("CUDA libraries were not found.");
}

std::vector<PlatformInfo> CudaCore::getPlatformInfo() const
{
    throw std::runtime_error("CUDA libraries were not found.");
}

std::vector<DeviceInfo> CudaCore::getDeviceInfo(const size_t platformIndex) const
{
    throw std::runtime_error("CUDA libraries were not found.");
}

void CudaCore::setCompilerOptions(const std::string& options)
{
    throw std::runtime_error("CUDA libraries were not found.");
}

void CudaCore::clearCache() const
{
    throw std::runtime_error("CUDA libraries were not found.");
}

KernelRunResult CudaCore::runKernel(const std::string& source, const std::string& kernelName, const std::vector<size_t>& globalSize,
    const std::vector<size_t>& localSize, const std::vector<KernelArgument>& arguments) const
{
    throw std::runtime_error("CUDA libraries were not found.");
}

#endif // USE_CUDA

} // namespace ktt
