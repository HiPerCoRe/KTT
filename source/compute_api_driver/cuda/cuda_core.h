#pragma once

#include <ostream>
#include <string>
#include <vector>

#include "../compute_api_driver.h"
#include "../../dto/device_info.h"
#include "../../dto/kernel_run_result.h"
#include "../../dto/platform_info.h"
#include "../../kernel_argument/kernel_argument.h"

namespace ktt
{

#ifdef USE_CUDA

// to do

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

private:
    size_t deviceIndex;
};

#endif // USE_CUDA

} // namespace ktt
