#pragma once

#include <ostream>
#include <string>
#include <vector>

#include "../dto/device_info.h"
#include "../dto/kernel_run_result.h"
#include "../dto/platform_info.h"
#include "../kernel_argument/kernel_argument.h"

namespace ktt
{

class ComputeApiDriver
{
public:
    // Destructor
    virtual ~ComputeApiDriver() = default;

    // Kernel execution method
    virtual KernelRunResult runKernel(const std::string& source, const std::string& kernelName, const std::vector<size_t>& globalSize,
        const std::vector<size_t>& localSize, const std::vector<const KernelArgument*>& argumentPointers) const = 0;

    // Compute API compiler options setup
    virtual void setCompilerOptions(const std::string& options) = 0;

    // Cache handling
    virtual void clearCache() const = 0;

    // Info retrieval methods
    virtual void printComputeApiInfo(std::ostream& outputTarget) const = 0;
    virtual std::vector<PlatformInfo> getPlatformInfo() const = 0;
    virtual std::vector<DeviceInfo> getDeviceInfo(const size_t platformIndex) const = 0;
};

} // namespace ktt
