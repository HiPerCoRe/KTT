#pragma once

#ifdef KTT_PLATFORM_OPENCL

#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <vector>
#include <api/user_initializer.h>
#include <compute_engine/opencl/opencl_buffer.h>
#include <compute_engine/opencl/opencl_command_queue.h>
#include <compute_engine/opencl/opencl_context.h>
#include <compute_engine/opencl/opencl_device.h>
#include <compute_engine/opencl/opencl_event.h>
#include <compute_engine/opencl/opencl_kernel.h>
#include <compute_engine/opencl/opencl_platform.h>
#include <compute_engine/opencl/opencl_program.h>
#include <compute_engine/compute_engine.h>

#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
#include <compute_engine/opencl/gpa/gpa_interface.h>
#include <compute_engine/opencl/gpa/gpa_profiling_context.h>
#include <compute_engine/opencl/gpa/gpa_profiling_instance.h>
#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY

namespace ktt
{

class OpenCLEngine : public ComputeEngine
{
public:
    explicit OpenCLEngine(const PlatformIndex platformIndex, const DeviceIndex deviceIndex, const uint32_t queueCount);
    explicit OpenCLEngine(const UserInitializer& initializer);

    // Kernel handling methods
    KernelResult runKernel(const KernelRuntimeData& kernelData, const std::vector<KernelArgument*>& argumentPointers,
        const std::vector<OutputDescriptor>& outputDescriptors) override;
    EventId runKernelAsync(const KernelRuntimeData& kernelData, const std::vector<KernelArgument*>& argumentPointers, const QueueId queue) override;
    KernelResult getKernelResult(const EventId id, const std::vector<OutputDescriptor>& outputDescriptors) const override;
    uint64_t getKernelOverhead(const EventId id) const override;

    // Utility methods
    void setCompilerOptions(const std::string& options) override;
    void setGlobalSizeType(const GlobalSizeType type) override;
    void setAutomaticGlobalSizeCorrection(const bool flag) override;
    void setKernelCacheUsage(const bool flag) override;
    void setKernelCacheCapacity(const size_t capacity) override;
    void clearKernelCache() override;

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
    void resizeArgument(const ArgumentId id, const size_t newSize, const bool preserveData) override;
    void getArgumentHandle(const ArgumentId id, BufferMemory& handle) override;
    void addUserBuffer(UserBuffer buffer, KernelArgument& kernelArgument) override;
    void setPersistentBufferUsage(const bool flag) override;
    void clearBuffer(const ArgumentId id) override;
    void clearBuffers() override;
    void clearBuffers(const ArgumentAccessType accessType) override;

    // Information retrieval methods
    void printComputeAPIInfo(std::ostream& outputTarget) const override;
    std::vector<PlatformInfo> getPlatformInfo() const override;
    std::vector<DeviceInfo> getDeviceInfo(const PlatformIndex platform) const override;
    DeviceInfo getCurrentDeviceInfo() const override;

    // Kernel profiling methods
    void initializeKernelProfiling(const KernelRuntimeData& kernelData) override;
    EventId runKernelWithProfiling(const KernelRuntimeData& kernelData, const std::vector<KernelArgument*>& argumentPointers,
        const QueueId queue) override;
    uint64_t getRemainingKernelProfilingRuns(const std::string& kernelName, const std::string& kernelSource) override;
    bool hasAccurateRemainingKernelProfilingRuns() const override;
    KernelResult getKernelResultWithProfiling(const EventId id, const std::vector<OutputDescriptor>& outputDescriptors) override;
    void setKernelProfilingCounters(const std::vector<std::string>& counterNames) override;

    std::unique_ptr<OpenCLProgram> createAndBuildProgram(const std::string& source) const;

private:
    // Attributes
    PlatformIndex platformIndex;
    DeviceIndex deviceIndex;
    std::string compilerOptions;
    GlobalSizeType globalSizeType;
    bool globalSizeCorrection;
    bool kernelCacheFlag;
    size_t kernelCacheCapacity;
    bool persistentBufferFlag;
    mutable EventId nextEventId;
    std::unique_ptr<OpenCLContext> context;
    std::vector<std::unique_ptr<OpenCLCommandQueue>> commandQueues;
    std::set<std::unique_ptr<OpenCLBuffer>> buffers;
    std::set<std::unique_ptr<OpenCLBuffer>> persistentBuffers;
    std::set<std::unique_ptr<OpenCLBuffer>> userBuffers;
    std::map<std::pair<std::string, std::string>, std::pair<std::unique_ptr<OpenCLKernel>, std::unique_ptr<OpenCLProgram>>> kernelCache;
    mutable std::map<EventId, std::unique_ptr<OpenCLEvent>> kernelEvents;
    mutable std::map<EventId, std::unique_ptr<OpenCLEvent>> bufferEvents;

    #if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
    std::unique_ptr<GPAInterface> gpaInterface;
    std::unique_ptr<GPAProfilingContext> gpaProfilingContext;
    std::map<std::pair<std::string, std::string>, std::vector<EventId>> kernelToEventMap;
    std::map<std::pair<std::string, std::string>, std::unique_ptr<GPAProfilingInstance>> kernelProfilingInstances;
    #endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY

    // Helper methods
    void initializeProfiler();
    void setKernelArgument(OpenCLKernel& kernel, KernelArgument& argument);
    void setKernelArgument(OpenCLKernel& kernel, KernelArgument& argument, const std::vector<LocalMemoryModifier>& modifiers);
    EventId enqueueKernel(OpenCLKernel& kernel, const std::vector<size_t>& globalSize, const std::vector<size_t>& localSize,
        const QueueId queue, const uint64_t kernelLaunchOverhead) const;
    KernelResult createKernelResult(const EventId id) const;
    static PlatformInfo getOpenCLPlatformInfo(const PlatformIndex platform);
    static DeviceInfo getOpenCLDeviceInfo(const PlatformIndex platform, const DeviceIndex device);
    static std::vector<OpenCLPlatform> getOpenCLPlatforms();
    static std::vector<OpenCLDevice> getOpenCLDevices(const OpenCLPlatform& platform);
    static DeviceType getDeviceType(const cl_device_type deviceType);
    OpenCLBuffer* findBuffer(const ArgumentId id) const;
    void setKernelArgumentVector(OpenCLKernel& kernel, const OpenCLBuffer& buffer) const;
    bool loadBufferFromCache(const ArgumentId id, OpenCLKernel& kernel) const;
    void checkLocalMemoryModifiers(const std::vector<KernelArgument*>& argumentPointers, const std::vector<LocalMemoryModifier>& modifiers) const;

    #if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
    void initializeKernelProfiling(const std::string& kernelName, const std::string& kernelSource);
    const std::pair<std::string, std::string>& getKernelFromEvent(const EventId id) const;
    static const std::vector<std::string>& getDefaultGPAProfilingCounters();
    void launchDummyPass(const std::string& kernelName, const std::string& kernelSource);
    #endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY
};

} // namespace ktt

#endif // KTT_PLATFORM_OPENCL
