#pragma once

#include <memory>
#include <ostream>
#include <string>
#include <vector>

#ifdef USE_CUDA
#include "cuda.h"
#include "nvrtc.h"

#include "cuda_device.h"
#include "cuda_utility.h"
#endif // USE_CUDA

#include "../compute_api_driver.h"
#include "../../dto/device_info.h"
#include "../../dto/kernel_run_result.h"
#include "../../dto/platform_info.h"
#include "../../kernel_argument/kernel_argument.h"

namespace ktt
{

#ifdef USE_CUDA

class CudaCore : public ComputeApiDriver
{
public:
    // Constructor
    explicit CudaCore(const size_t deviceIndex);

    // Platform and device retrieval methods
    virtual void printComputeApiInfo(std::ostream& outputTarget) const override;
    virtual std::vector<PlatformInfo> getPlatformInfo() const override;
    virtual std::vector<DeviceInfo> getDeviceInfo(const size_t platformIndex) const override;

    // Compiler options setup
    virtual void setCompilerOptions(const std::string& options) override;

    // Cache handling
    virtual void clearCache() const override;

    // High-level kernel execution methods
    virtual KernelRunResult runKernel(const std::string& source, const std::string& kernelName, const std::vector<size_t>& globalSize,
        const std::vector<size_t>& localSize, const std::vector<KernelArgument>& arguments) const override;

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

    // Compiler options setup
    virtual void setCompilerOptions(const std::string& options) override;

    // Cache handling
    virtual void clearCache() const override;

    // High-level kernel execution methods
    virtual KernelRunResult runKernel(const std::string& source, const std::string& kernelName, const std::vector<size_t>& globalSize,
        const std::vector<size_t>& localSize, const std::vector<KernelArgument>& arguments) const override;
};

#endif // USE_CUDA

} // namespace ktt
