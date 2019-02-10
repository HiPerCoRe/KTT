#pragma once

#ifdef KTT_PLATFORM_CUDA

#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <cuda.h>
#include <nvrtc.h>
#include <compute_engine/cuda/cuda_buffer.h>
#include <compute_engine/cuda/cuda_context.h>
#include <compute_engine/cuda/cuda_device.h>
#include <compute_engine/cuda/cuda_event.h>
#include <compute_engine/cuda/cuda_kernel.h>
#include <compute_engine/cuda/cuda_program.h>
#include <compute_engine/cuda/cuda_stream.h>
#include <compute_engine/cuda/cuda_utility.h>
#include <compute_engine/compute_engine.h>
#ifdef KTT_PROFILING
#include <cupti.h>
#include <compute_engine/cuda/cuda_profiling_state.h>
#endif // KTT_PROFILING

namespace ktt
{

class CUDAEngine : public ComputeEngine
{
public:
    // Constructor
    explicit CUDAEngine(const DeviceIndex deviceIndex, const uint32_t queueCount);

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
    KernelResult getKernelResultWithProfiling(const EventId id, const std::vector<OutputDescriptor>& outputDescriptors) override;
    void setKernelProfilingCounters(const std::vector<std::string>& counterNames) override;

private:
    DeviceIndex deviceIndex;
    uint32_t queueCount;
    std::string compilerOptions;
    GlobalSizeType globalSizeType;
    bool globalSizeCorrection;
    bool kernelCacheFlag;
    size_t kernelCacheCapacity;
    bool persistentBufferFlag;
    mutable EventId nextEventId;
    std::unique_ptr<CUDAContext> context;
    std::vector<std::unique_ptr<CUDAStream>> streams;
    std::set<std::unique_ptr<CUDABuffer>> buffers;
    std::set<std::unique_ptr<CUDABuffer>> persistentBuffers;
    std::map<std::pair<std::string, std::string>, std::unique_ptr<CUDAKernel>> kernelCache;
    mutable std::map<EventId, std::pair<std::unique_ptr<CUDAEvent>, std::unique_ptr<CUDAEvent>>> kernelEvents;
    mutable std::map<EventId, std::pair<std::unique_ptr<CUDAEvent>, std::unique_ptr<CUDAEvent>>> bufferEvents;
#ifdef KTT_PROFILING
    std::vector<std::pair<std::string, CUpti_MetricID>> profilingMetrics;
    std::map<std::pair<std::string, std::string>, std::vector<EventId>> kernelToEventMap;
    std::map<std::pair<std::string, std::string>, CUDAProfilingState> kernelProfilingStates;
#endif // KTT_PROFILING

    std::unique_ptr<CUDAProgram> createAndBuildProgram(const std::string& source) const;
    EventId enqueueKernel(CUDAKernel& kernel, const std::vector<size_t>& globalSize, const std::vector<size_t>& localSize,
        const std::vector<CUdeviceptr*>& kernelArguments, const size_t localMemorySize, const QueueId queue, const uint64_t kernelLaunchOverhead);
    KernelResult createKernelResult(const EventId id) const;
    DeviceInfo getCUDADeviceInfo(const DeviceIndex deviceIndex) const;
    std::vector<CUDADevice> getCUDADevices() const;
    std::vector<CUdeviceptr*> getKernelArguments(const std::vector<KernelArgument*>& argumentPointers);
    size_t getSharedMemorySizeInBytes(const std::vector<KernelArgument*>& argumentPointers, const std::vector<LocalMemoryModifier>& modifiers) const;
    CUDABuffer* findBuffer(const ArgumentId id) const;
    CUdeviceptr* loadBufferFromCache(const ArgumentId id) const;

#ifdef KTT_PROFILING
    void initializeKernelProfiling(const std::string& kernelName, const std::string& kernelSource);
    const std::pair<std::string, std::string>& getKernelFromEvent(const EventId id) const;
    CUpti_MetricID getMetricIdFromName(const std::string& metricName);
    std::vector<std::pair<std::string, CUpti_MetricID>> getProfilingMetricsForCurrentDevice(const std::vector<std::string>& metricNames);
    static void CUPTIAPI getMetricValueCallback(void* userdata, CUpti_CallbackDomain domain, CUpti_CallbackId id, const CUpti_CallbackData* info);
    static const std::vector<std::string>& getDefaultProfilingMetricNames();
#endif // KTT_PROFILING
};

} // namespace ktt

#endif // KTT_PLATFORM_CUDA
