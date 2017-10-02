#pragma once

#include <ostream>
#include <string>
#include <vector>

#include "api/device_info.h"
#include "api/platform_info.h"
#include "dto/argument_output_descriptor.h"
#include "dto/kernel_run_result.h"
#include "dto/kernel_runtime_data.h"
#include "kernel_argument/kernel_argument.h"

namespace ktt
{

class ComputeEngine
{
public:
    // Destructor
    virtual ~ComputeEngine() = default;

    // Kernel execution method
    virtual KernelRunResult runKernel(const KernelRuntimeData& kernelData, const std::vector<KernelArgument*>& argumentPointers,
        const std::vector<ArgumentOutputDescriptor>& outputDescriptors) = 0;

    // Compute API compiler options setup
    virtual void setCompilerOptions(const std::string& options) = 0;

    // Argument handling methods
    virtual void uploadArgument(KernelArgument& kernelArgument) = 0;
    virtual void updateArgument(const size_t argumentId, const void* data, const size_t dataSizeInBytes) = 0;
    virtual KernelArgument downloadArgument(const size_t argumentId) const = 0;
    virtual void downloadArgument(const size_t argumentId, void* destination, const size_t dataSizeInBytes) const = 0;
    virtual void clearBuffer(const size_t argumentId) = 0;
    virtual void clearBuffers() = 0;
    virtual void clearBuffers(const ArgumentAccessType& accessType) = 0;

    // Information retrieval methods
    virtual void printComputeApiInfo(std::ostream& outputTarget) const = 0;
    virtual std::vector<PlatformInfo> getPlatformInfo() const = 0;
    virtual std::vector<DeviceInfo> getDeviceInfo(const size_t platformIndex) const = 0;
    virtual DeviceInfo getCurrentDeviceInfo() const = 0;
};

} // namespace ktt
