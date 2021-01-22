#pragma once

#ifdef KTT_API_CUDA

#include <map>
#include <memory>
#include <vector>

#include <Api/ComputeApiInitializer.h>
#include <ComputeEngine/Cuda/Actions/CudaComputeAction.h>
#include <ComputeEngine/Cuda/Actions/CudaTransferAction.h>
#include <ComputeEngine/Cuda/Buffers/CudaBuffer.h>
#include <ComputeEngine/Cuda/CudaContext.h>
#include <ComputeEngine/Cuda/CudaKernel.h>
#include <ComputeEngine/Cuda/CudaStream.h>
#include <ComputeEngine/ActionIdGenerator.h>
#include <ComputeEngine/ComputeEngine.h>
#include <Utility/LruCache.h>

#ifdef KTT_PROFILING_CUPTI_LEGACY
#include <ComputeEngine/Cuda/CuptiLegacy/CuptiInstance.h>
#elif KTT_PROFILING_CUPTI
//#include <compute_engine/cuda/cupti/cupti_metric_interface.h>
//#include <compute_engine/cuda/cupti/cupti_profiler.h>
//#include <compute_engine/cuda/cupti/cupti_profiling_instance.h>
#endif // KTT_PROFILING_CUPTI

namespace ktt
{

class CudaEngine : public ComputeEngine
{
public:
    explicit CudaEngine(const DeviceIndex deviceIndex, const uint32_t queueCount);
    explicit CudaEngine(const ComputeApiInitializer& initializer);

    // Kernel methods
    ComputeActionId RunKernelAsync(const KernelComputeData& data, const QueueId queueId) override;
    KernelResult WaitForComputeAction(const ComputeActionId id) override;

    // Profiling methods
    KernelResult RunKernelWithProfiling(const KernelComputeData& data, const QueueId queueId) override;
    void SetProfilingCounters(const std::vector<std::string>& counters) override;
    bool IsProfilingSessionActive(const KernelComputeId& id) override;
    uint64_t GetRemainingProfilingRuns(const KernelComputeId& id) override;
    bool HasAccurateRemainingProfilingRuns() const override;

    // Buffer methods
    TransferActionId UploadArgument(KernelArgument& kernelArgument, const QueueId queueId) override;
    TransferActionId UpdateArgument(const ArgumentId id, const QueueId queueId, const void* data,
        const size_t dataSize) override;
    TransferActionId DownloadArgument(const ArgumentId id, const QueueId queueId, void* destination,
        const size_t dataSize) override;
    TransferActionId CopyArgument(const ArgumentId destination, const QueueId queueId, const ArgumentId source,
        const size_t dataSize) override;
    TransferResult WaitForTransferAction(const TransferActionId id) override;
    void ResizeArgument(const ArgumentId id, const size_t newSize, const bool preserveData) override;
    void GetUnifiedMemoryBufferHandle(const ArgumentId id, UnifiedBufferMemory& handle) override;
    void AddCustomBuffer(KernelArgument& kernelArgument, ComputeBuffer buffer) override;
    void ClearBuffer(const ArgumentId id) override;
    void ClearBuffers() override;

    // Queue methods
    QueueId GetDefaultQueue() const override;
    std::vector<QueueId> GetAllQueues() const override;
    void SynchronizeQueue(const QueueId queueId) override;
    void SynchronizeDevice() override;

    // Information retrieval methods
    std::vector<PlatformInfo> GetPlatformInfo() const override;
    std::vector<DeviceInfo> GetDeviceInfo(const PlatformIndex platformIndex) const override;
    DeviceInfo GetCurrentDeviceInfo() const override;

    // Utility methods
    void SetCompilerOptions(const std::string& options) override;
    void SetGlobalSizeType(const GlobalSizeType type) override;
    void SetAutomaticGlobalSizeCorrection(const bool flag) override;
    void SetKernelCacheCapacity(const uint64_t capacity) override;
    void ClearKernelCache() override;

private:
    DeviceIndex m_DeviceIndex;
    ActionIdGenerator m_Generator;
    std::unique_ptr<CudaContext> m_Context;
    std::vector<std::unique_ptr<CudaStream>> m_Streams;
    std::map<ArgumentId, std::unique_ptr<CudaBuffer>> m_Buffers;
    LruCache<KernelComputeId, std::shared_ptr<CudaKernel>> m_KernelCache;
    std::map<ComputeActionId, std::unique_ptr<CudaComputeAction>> m_ComputeActions;
    std::map<TransferActionId, std::unique_ptr<CudaTransferAction>> m_TransferActions;

    #ifdef KTT_PROFILING_CUPTI_LEGACY
    std::map<KernelComputeId, std::unique_ptr<CuptiInstance>> m_CuptiInstances;
    #elif KTT_PROFILING_CUPTI
    //std::unique_ptr<CUPTIProfiler> profiler;
    //std::unique_ptr<CUPTIMetricInterface> metricInterface;
    //std::vector<std::string> profilingCounters;
    //std::map<std::pair<std::string, std::string>, std::vector<EventId>> kernelToEventMap;
    //std::map<std::pair<std::string, std::string>, std::unique_ptr<CUPTIProfilingInstance>> kernelProfilingInstances;
    #endif // KTT_PROFILING_CUPTI

    //void initializeCompilerOptions();
    //void initializeProfiler();
    //std::unique_ptr<CUDAProgram> createAndBuildProgram(const std::string& source) const;
    //EventId enqueueKernel(CUDAKernel& kernel, const std::vector<size_t>& globalSize, const std::vector<size_t>& localSize,
    //    const std::vector<CUdeviceptr*>& kernelArguments, const size_t localMemorySize, const QueueId queue, const uint64_t kernelLaunchOverhead);
    //KernelResult createKernelResult(const EventId id) const;
    //std::vector<CUdeviceptr*> getKernelArguments(const std::vector<KernelArgument*>& argumentPointers);
    //size_t getSharedMemorySizeInBytes(const std::vector<KernelArgument*>& argumentPointers, const std::vector<LocalMemoryModifier>& modifiers) const;
    //CUDABuffer* findBuffer(const ArgumentId id) const;
    //CUdeviceptr* loadBufferFromCache(const ArgumentId id) const;

    //#ifdef KTT_PROFILING_CUPTI_LEGACY
    //void initializeKernelProfiling(const std::string& kernelName, const std::string& kernelSource);
    //const std::pair<std::string, std::string>& getKernelFromEvent(const EventId id) const;
    //CUpti_MetricID getMetricIdFromName(const std::string& metricName);
    //std::vector<std::pair<std::string, CUpti_MetricID>> getProfilingMetricsForCurrentDevice(const std::vector<std::string>& metricNames);
    //#elif KTT_PROFILING_CUPTI
    //void initializeKernelProfiling(const std::string& kernelName, const std::string& kernelSource);
    //const std::pair<std::string, std::string>& getKernelFromEvent(const EventId id) const;
    //static const std::vector<std::string>& getDefaultProfilingCounters();
    //#endif // KTT_PROFILING_CUPTI
};

} // namespace ktt

#endif // KTT_API_CUDA
