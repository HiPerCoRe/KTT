#pragma once

#include <ostream>
#include <string>
#include <vector>
#include "ktt_types.h"
#include "api/device_info.h"
#include "api/output_descriptor.h"
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

    // Kernel handling methods
    virtual KernelResult runKernel(const KernelRuntimeData& kernelData, const std::vector<KernelArgument*>& argumentPointers,
        const std::vector<OutputDescriptor>& outputDescriptors) = 0;
    virtual EventId runKernelAsync(const KernelRuntimeData& kernelData, const std::vector<KernelArgument*>& argumentPointers, const QueueId queue) = 0;
    virtual KernelResult getKernelResult(const EventId id, const std::vector<OutputDescriptor>& outputDescriptors) const = 0;

    // Utility methods
    virtual void setCompilerOptions(const std::string& options) = 0;
    virtual void setGlobalSizeType(const GlobalSizeType type) = 0;
    virtual void setAutomaticGlobalSizeCorrection(const bool flag) = 0;
    virtual void setProgramCache(const bool flag) = 0;
    virtual void clearProgramCache() = 0;

    // Queue handling methods
    virtual QueueId getDefaultQueue() const = 0;
    virtual std::vector<QueueId> getAllQueues() const = 0;
    virtual void synchronizeQueue(const QueueId queue) = 0;
    virtual void synchronizeDevice() = 0;
    virtual void clearEvents() = 0;

    // Argument handling methods
    virtual uint64_t uploadArgument(KernelArgument& kernelArgument) = 0;
    virtual EventId uploadArgumentAsync(KernelArgument& kernelArgument, const QueueId queue) = 0;
    virtual uint64_t updateArgument(const ArgumentId id, const void* data, const size_t dataSizeInBytes) = 0;
    virtual EventId updateArgumentAsync(const ArgumentId id, const void* data, const size_t dataSizeInBytes, const QueueId queue) = 0;
    virtual uint64_t downloadArgument(const ArgumentId id, void* destination, const size_t dataSizeInBytes) const = 0;
    virtual EventId downloadArgumentAsync(const ArgumentId id, void* destination, const size_t dataSizeInBytes, const QueueId queue) const = 0;
    virtual KernelArgument downloadArgumentObject(const ArgumentId id, uint64_t* downloadDuration) const = 0;
    //virtual uint64_t copyBuffer(const ArgumentId destination, const ArgumentId source, const size_t dataSizeInBytes) = 0;
    //virtual uint64_t copyBufferAsync(const ArgumentId destination, const ArgumentId source, const size_t dataSizeInBytes, const QueueId queue) = 0;
    virtual uint64_t getArgumentOperationDuration(const EventId id) const = 0;
    virtual void clearBuffer(const ArgumentId id) = 0;
    virtual void clearBuffers() = 0;
    virtual void clearBuffers(const ArgumentAccessType accessType) = 0;

    // Information retrieval methods
    virtual void printComputeAPIInfo(std::ostream& outputTarget) const = 0;
    virtual std::vector<PlatformInfo> getPlatformInfo() const = 0;
    virtual std::vector<DeviceInfo> getDeviceInfo(const PlatformIndex platform) const = 0;
    virtual DeviceInfo getCurrentDeviceInfo() const = 0;
};

} // namespace ktt
