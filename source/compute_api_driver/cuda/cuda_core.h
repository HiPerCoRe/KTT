#pragma once

#include <memory>
#include <ostream>
#include <string>
#include <vector>

#ifdef PLATFORM_CUDA
#include "cuda.h"
#include "nvrtc.h"

#include "cuda_device.h"
#include "cuda_utility.h"
#endif // PLATFORM_CUDA

#include "../compute_api_driver.h"
#include "../../dto/device_info.h"
#include "../../dto/kernel_run_result.h"
#include "../../dto/platform_info.h"
#include "../../kernel_argument/kernel_argument.h"

namespace ktt
{

#ifdef PLATFORM_CUDA

class CudaCore : public ComputeApiDriver
{
public:
    // Constructor
    explicit CudaCore(const size_t deviceIndex);

    // Platform and device retrieval methods
    virtual void printComputeApiInfo(std::ostream& outputTarget) const override;
    virtual std::vector<PlatformInfo> getPlatformInfo() const override;
    virtual std::vector<DeviceInfo> getDeviceInfo(const size_t platformIndex) const override;
    virtual DeviceInfo getCurrentDeviceInfo() const override;

    // Compiler options setup
    virtual void setCompilerOptions(const std::string& options) override;

    // Argument cache handling
    virtual void setCacheUsage(const bool flag, const ArgumentMemoryType& argumentMemoryType) override;
    virtual void clearCache() override;
    virtual void clearCache(const ArgumentMemoryType& argumentMemoryType) override;

    // High-level kernel execution methods
    virtual KernelRunResult runKernel(const std::string& source, const std::string& kernelName, const std::vector<size_t>& globalSize,
        const std::vector<size_t>& localSize, const std::vector<const KernelArgument*>& argumentPointers) override;

private:
    size_t deviceIndex;
    std::string compilerOptions;

    std::vector<CudaDevice> getCudaDevices() const;
};

#else

class CudaCore : public ComputeApiDriver
{
public:
    // Constructor
    explicit CudaCore(const size_t deviceIndex);

    // Platform and device retrieval methods
    virtual void printComputeApiInfo(std::ostream& outputTarget) const override;
    virtual std::vector<PlatformInfo> getPlatformInfo() const override;
    virtual std::vector<DeviceInfo> getDeviceInfo(const size_t platformIndex) const override;
    virtual DeviceInfo getCurrentDeviceInfo() const override;

    // Compiler options setup
    virtual void setCompilerOptions(const std::string& options) override;

    // Argument cache handling
    virtual void setCacheUsage(const bool flag, const ArgumentMemoryType& argumentMemoryType) override;
    virtual void clearCache() override;
    virtual void clearCache(const ArgumentMemoryType& argumentMemoryType) override;

    // High-level kernel execution methods
    virtual KernelRunResult runKernel(const std::string& source, const std::string& kernelName, const std::vector<size_t>& globalSize,
        const std::vector<size_t>& localSize, const std::vector<const KernelArgument*>& argumentPointers) override;
};

#endif // PLATFORM_CUDA

} // namespace ktt
