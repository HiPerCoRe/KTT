#pragma once

#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <vector>
#include "compute_engine/compute_engine.h"

#ifdef PLATFORM_OPENCL
#include "opencl_buffer.h"
#include "opencl_command_queue.h"
#include "opencl_context.h"
#include "opencl_device.h"
#include "opencl_event.h"
#include "opencl_kernel.h"
#include "opencl_platform.h"
#include "opencl_program.h"
#endif // PLATFORM_OPENCL

namespace ktt
{

#ifdef PLATFORM_OPENCL

class OpenCLEngine : public ComputeEngine
{
public:
    // Constructor
    explicit OpenCLEngine(const PlatformIndex platformIndex, const DeviceIndex deviceIndex, const uint32_t queueCount);

    // Kernel handling methods
    KernelResult runKernel(const KernelRuntimeData& kernelData, const std::vector<KernelArgument*>& argumentPointers,
        const std::vector<OutputDescriptor>& outputDescriptors) override;
    EventId runKernelAsync(const KernelRuntimeData& kernelData, const std::vector<KernelArgument*>& argumentPointers, const QueueId queue) override;
    KernelResult getKernelResult(const EventId id, const std::vector<OutputDescriptor>& outputDescriptors) const override;

    // Utility methods
    void setCompilerOptions(const std::string& options) override;
    void setGlobalSizeType(const GlobalSizeType type) override;
    void setAutomaticGlobalSizeCorrection(const bool flag) override;
    void setProgramCache(const bool flag) override;
    void clearProgramCache() override;

    // Queue handling methods
    QueueId getDefaultQueue() const override;
    std::vector<QueueId> getAllQueues() const override;
    void synchronizeQueue(const QueueId queue) override;
    void synchronizeDevice() override;
    void clearEvents() override;

    // Argument handling methods
    uint64_t uploadArgument(KernelArgument& kernelArgument) override;
    EventId uploadArgumentAsync(KernelArgument& kernelArgument, const QueueId queue) override;
    uint64_t updateArgument(const ArgumentId id, const void* data, const size_t dataSizeInBytes) override;
    EventId updateArgumentAsync(const ArgumentId id, const void* data, const size_t dataSizeInBytes, const QueueId queue) override;
    uint64_t downloadArgument(const ArgumentId id, void* destination, const size_t dataSizeInBytes) const override;
    EventId downloadArgumentAsync(const ArgumentId id, void* destination, const size_t dataSizeInBytes, const QueueId queue) const override;
    KernelArgument downloadArgumentObject(const ArgumentId id, uint64_t* downloadDuration) const override;
    uint64_t copyArgument(const ArgumentId destination, const ArgumentId source, const size_t dataSizeInBytes) override;
    EventId copyArgumentAsync(const ArgumentId destination, const ArgumentId source, const size_t dataSizeInBytes, const QueueId queue) override;
    uint64_t persistArgument(KernelArgument& kernelArgument, const bool flag) override;
    uint64_t getArgumentOperationDuration(const EventId id) const override;
    void clearBuffer(const ArgumentId id) override;
    void clearBuffers() override;
    void clearBuffers(const ArgumentAccessType accessType) override;

    // Information retrieval methods
    void printComputeAPIInfo(std::ostream& outputTarget) const override;
    std::vector<PlatformInfo> getPlatformInfo() const override;
    std::vector<DeviceInfo> getDeviceInfo(const PlatformIndex platform) const override;
    DeviceInfo getCurrentDeviceInfo() const override;

    // Low-level kernel execution methods
    std::unique_ptr<OpenCLProgram> createAndBuildProgram(const std::string& source) const;
    void setKernelArgument(OpenCLKernel& kernel, KernelArgument& argument);
    void setKernelArgument(OpenCLKernel& kernel, KernelArgument& argument, const std::vector<LocalMemoryModifier>& modifiers);
    EventId enqueueKernel(OpenCLKernel& kernel, const std::vector<size_t>& globalSize, const std::vector<size_t>& localSize,
        const QueueId queue, const uint64_t kernelLaunchOverhead) const;

private:
    // Attributes
    PlatformIndex platformIndex;
    DeviceIndex deviceIndex;
    uint32_t queueCount;
    std::string compilerOptions;
    GlobalSizeType globalSizeType;
    bool globalSizeCorrection;
    bool programCacheFlag;
    mutable EventId nextEventId;
    std::unique_ptr<OpenCLContext> context;
    std::vector<std::unique_ptr<OpenCLCommandQueue>> commandQueues;
    std::set<std::unique_ptr<OpenCLBuffer>> buffers;
    std::set<std::unique_ptr<OpenCLBuffer>> persistentBuffers;
    std::map<std::string, std::unique_ptr<OpenCLProgram>> programCache;
    mutable std::map<EventId, std::unique_ptr<OpenCLEvent>> kernelEvents;
    mutable std::map<EventId, std::unique_ptr<OpenCLEvent>> bufferEvents;

    // Helper methods
    static PlatformInfo getOpenCLPlatformInfo(const PlatformIndex platform);
    static DeviceInfo getOpenCLDeviceInfo(const PlatformIndex platform, const DeviceIndex device);
    static std::vector<OpenCLPlatform> getOpenCLPlatforms();
    static std::vector<OpenCLDevice> getOpenCLDevices(const OpenCLPlatform& platform);
    static DeviceType getDeviceType(const cl_device_type deviceType);
    OpenCLBuffer* findBuffer(const ArgumentId id) const;
    void setKernelArgumentVector(OpenCLKernel& kernel, const OpenCLBuffer& buffer) const;
    bool loadBufferFromCache(const ArgumentId id, OpenCLKernel& kernel) const;
    void checkLocalMemoryModifiers(const std::vector<KernelArgument*>& argumentPointers, const std::vector<LocalMemoryModifier>& modifiers) const;
};

#else

class OpenCLEngine : public ComputeEngine
{
public:
    // Constructor
    explicit OpenCLEngine(const PlatformIndex platformIndex, const DeviceIndex deviceIndex, const uint32_t queueCount);

    // Kernel handling methods
    KernelResult runKernel(const KernelRuntimeData& kernelData, const std::vector<KernelArgument*>& argumentPointers,
        const std::vector<OutputDescriptor>& outputDescriptors) override;
    EventId runKernelAsync(const KernelRuntimeData& kernelData, const std::vector<KernelArgument*>& argumentPointers, const QueueId queue) override;
    KernelResult getKernelResult(const EventId id, const std::vector<OutputDescriptor>& outputDescriptors) const override;

    // Utility methods
    void setCompilerOptions(const std::string& options) override;
    void setGlobalSizeType(const GlobalSizeType type) override;
    void setAutomaticGlobalSizeCorrection(const bool flag) override;
    void setProgramCache(const bool flag) override;
    void clearProgramCache() override;

    // Queue handling methods
    QueueId getDefaultQueue() const override;
    std::vector<QueueId> getAllQueues() const override;
    void synchronizeQueue(const QueueId queue) override;
    void synchronizeDevice() override;
    void clearEvents() override;

    // Argument handling methods
    uint64_t uploadArgument(KernelArgument& kernelArgument) override;
    EventId uploadArgumentAsync(KernelArgument& kernelArgument, const QueueId queue) override;
    uint64_t updateArgument(const ArgumentId id, const void* data, const size_t dataSizeInBytes) override;
    EventId updateArgumentAsync(const ArgumentId id, const void* data, const size_t dataSizeInBytes, const QueueId queue) override;
    uint64_t downloadArgument(const ArgumentId id, void* destination, const size_t dataSizeInBytes) const override;
    EventId downloadArgumentAsync(const ArgumentId id, void* destination, const size_t dataSizeInBytes, const QueueId queue) const override;
    KernelArgument downloadArgumentObject(const ArgumentId id, uint64_t* downloadDuration) const override;
    uint64_t copyArgument(const ArgumentId destination, const ArgumentId source, const size_t dataSizeInBytes) override;
    EventId copyArgumentAsync(const ArgumentId destination, const ArgumentId source, const size_t dataSizeInBytes, const QueueId queue) override;
    uint64_t persistArgument(KernelArgument& kernelArgument, const bool flag) override;
    uint64_t getArgumentOperationDuration(const EventId id) const override;
    void clearBuffer(const ArgumentId id) override;
    void clearBuffers() override;
    void clearBuffers(const ArgumentAccessType accessType) override;

    // Information retrieval methods
    void printComputeAPIInfo(std::ostream& outputTarget) const override;
    std::vector<PlatformInfo> getPlatformInfo() const override;
    std::vector<DeviceInfo> getDeviceInfo(const PlatformIndex platform) const override;
    DeviceInfo getCurrentDeviceInfo() const override;
};

#endif // PLATFORM_OPENCL

} // namespace ktt
