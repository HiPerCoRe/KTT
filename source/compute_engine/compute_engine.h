#pragma once

#include <ostream>
#include <string>
#include <vector>
#include "ktt_types.h"
#include "api/argument_output_descriptor.h"
#include "api/device_info.h"
#include "api/platform_info.h"
#include "dto/kernel_result.h"
#include "dto/kernel_runtime_data.h"
#include "enum/global_size_type.h"
#include "kernel_argument/kernel_argument.h"

namespace ktt
{

class ComputeEngine
{
public:
    // Destructor
    virtual ~ComputeEngine() = default;

    // Kernel execution method
    virtual KernelResult runKernel(const QueueId queue, const KernelRuntimeData& kernelData, const std::vector<KernelArgument*>& argumentPointers,
        const std::vector<ArgumentOutputDescriptor>& outputDescriptors) = 0;

    // Utility methods
    virtual void setCompilerOptions(const std::string& options) = 0;
    virtual void setGlobalSizeType(const GlobalSizeType& type) = 0;
    virtual void setAutomaticGlobalSizeCorrection(const bool flag) = 0;

    // Queue handling methods
    virtual QueueId getDefaultQueue() const = 0;
    virtual QueueId createQueue() = 0;

    // Argument handling methods
    virtual void uploadArgument(const QueueId queue, KernelArgument& kernelArgument) = 0;
    virtual void updateArgument(const QueueId queue, const ArgumentId id, const void* data, const size_t dataSizeInBytes) = 0;
    virtual KernelArgument downloadArgument(const QueueId queue, const ArgumentId id) const = 0;
    virtual void downloadArgument(const QueueId queue, const ArgumentId id, void* destination) const = 0;
    virtual void downloadArgument(const QueueId queue, const ArgumentId id, void* destination, const size_t dataSizeInBytes) const = 0;
    virtual void clearBuffer(const ArgumentId id) = 0;
    virtual void clearBuffers() = 0;
    virtual void clearBuffers(const ArgumentAccessType& accessType) = 0;

    // Information retrieval methods
    virtual void printComputeApiInfo(std::ostream& outputTarget) const = 0;
    virtual std::vector<PlatformInfo> getPlatformInfo() const = 0;
    virtual std::vector<DeviceInfo> getDeviceInfo(const size_t platformIndex) const = 0;
    virtual DeviceInfo getCurrentDeviceInfo() const = 0;
};

} // namespace ktt
