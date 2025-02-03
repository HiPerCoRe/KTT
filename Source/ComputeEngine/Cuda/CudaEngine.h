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
#include <ComputeEngine/ComputeEngine.h>
#include <ComputeEngine/EngineConfiguration.h>
#include <Utility/IdGenerator.h>
#include <Utility/LruCache.h>

#ifdef KTT_PROFILING_CUPTI_LEGACY
#include <ComputeEngine/Cuda/CuptiLegacy/CuptiInstance.h>
#elif KTT_PROFILING_CUPTI
#include <ComputeEngine/Cuda/Cupti/CuptiInstance.h>
#include <ComputeEngine/Cuda/Cupti/CuptiMetricInterface.h>
#include <ComputeEngine/Cuda/Cupti/CuptiProfiler.h>
#endif // KTT_PROFILING_CUPTI

#ifdef KTT_POWER_USAGE_NVML
#include <ComputeEngine/Cuda/Nvml/NvmlPowerManager.h>
#endif // KTT_POWER_USAGE_NVML

namespace ktt
{

class CudaEngine : public ComputeEngine
{
public:
    explicit CudaEngine(const DeviceIndex deviceIndex, const uint32_t queueCount);
    explicit CudaEngine(const ComputeApiInitializer& initializer, std::vector<QueueId>& assignedQueueIds);

    // Kernel methods
    ComputeActionId RunKernelAsync(const KernelComputeData& data, const QueueId queueId, const bool powerMeasurementAllowed = true) override;
    ComputationResult WaitForComputeAction(const ComputeActionId id) override;
    void ClearData(const KernelComputeId& id) override;
    void ClearKernelData(const std::string& kernelName) override;

    // Profiling methods
    ComputationResult RunKernelWithProfiling(const KernelComputeData& data, const QueueId queueId) override;
    void SetProfilingCounters(const std::vector<std::string>& counters) override;
    bool IsProfilingSessionActive(const KernelComputeId& id) override;
    uint64_t GetRemainingProfilingRuns(const KernelComputeId& id) override;
    bool HasAccurateRemainingProfilingRuns() const override;
    bool SupportsMultiInstanceProfiling() const override;
    bool IsProfilingActive() const override;
    void SetProfiling(const bool profiling) override;

    // Buffer methods
    TransferActionId UploadArgument(KernelArgument& kernelArgument, const QueueId queueId) override;
    TransferActionId UpdateArgument(const ArgumentId& id, const QueueId queueId, const void* data,
        const size_t dataSize) override;
    TransferActionId DownloadArgument(const ArgumentId& id, const QueueId queueId, void* destination,
        const size_t dataSize) override;
    TransferActionId CopyArgument(const ArgumentId& destination, const QueueId queueId, const ArgumentId& source,
        const size_t dataSize) override;
    TransferResult WaitForTransferAction(const TransferActionId id) override;
    void ResizeArgument(const ArgumentId& id, const size_t newSize, const bool preserveData) override;
    void GetUnifiedMemoryBufferHandle(const ArgumentId& id, UnifiedBufferMemory& handle) override;
    void AddCustomBuffer(KernelArgument& kernelArgument, ComputeBuffer buffer) override;
    void ClearBuffer(const ArgumentId& id) override;
    void ClearBuffers() override;
    bool HasBuffer(const ArgumentId& id) override;

    // Queue methods
    QueueId AddComputeQueue(ComputeQueue queue) override;
    void RemoveComputeQueue(const QueueId id) override;
    QueueId GetDefaultQueue() const override;
    std::vector<QueueId> GetAllQueues() const override;
    void SynchronizeQueue(const QueueId queueId) override;
    void SynchronizeQueues() override;
    void SynchronizeDevice() override;

    // Information retrieval methods
    std::vector<PlatformInfo> GetPlatformInfo() const override;
    std::vector<DeviceInfo> GetDeviceInfo(const PlatformIndex platformIndex) const override;
    PlatformInfo GetCurrentPlatformInfo() const override;
    DeviceInfo GetCurrentDeviceInfo() const override;
    ComputeApi GetComputeApi() const override;
    GlobalSizeType GetGlobalSizeType() const override;

    // Utility methods
    void SetCompilerOptions(const std::string& options, const bool overrideDefault = false) override;
    void SetGlobalSizeType(const GlobalSizeType type) override;
    void SetAutomaticGlobalSizeCorrection(const bool flag) override;
    void SetKernelCacheCapacity(const uint64_t capacity) override;
    void ClearKernelCache() override;
    void EnsureThreadContext() override;

private:
    EngineConfiguration m_Configuration;
    DeviceIndex m_DeviceIndex;
    DeviceInfo m_DeviceInfo;
    IdGenerator<QueueId> m_QueueIdGenerator;
    IdGenerator<ComputeActionId> m_ComputeIdGenerator;
    IdGenerator<TransferActionId> m_TransferIdGenerator;
    std::unique_ptr<CudaContext> m_Context;
    std::map<QueueId, std::unique_ptr<CudaStream>> m_Streams;
    std::map<ArgumentId, std::unique_ptr<CudaBuffer>> m_Buffers;
    LruCache<KernelComputeId, std::shared_ptr<CudaKernel>> m_KernelCache;
    std::map<ComputeActionId, std::unique_ptr<CudaComputeAction>> m_ComputeActions;
    std::map<TransferActionId, std::unique_ptr<CudaTransferAction>> m_TransferActions;

#ifdef KTT_PROFILING_CUPTI_LEGACY
    std::map<KernelComputeId, std::unique_ptr<CuptiInstance>> m_CuptiInstances;
#elif KTT_PROFILING_CUPTI
    std::unique_ptr<CuptiProfiler> m_Profiler;
    std::unique_ptr<CuptiMetricInterface> m_MetricInterface;
    std::map<KernelComputeId, std::unique_ptr<CuptiInstance>> m_CuptiInstances;
#endif // KTT_PROFILING_CUPTI

#ifdef KTT_POWER_USAGE_NVML
    std::unique_ptr<NvmlPowerManager> m_PowerManager;
#endif // KTT_POWER_USAGE_NVML

    std::shared_ptr<CudaKernel> LoadKernel(const KernelComputeData& data);
    std::vector<CUdeviceptr*> GetKernelArguments(const std::vector<KernelArgument*>& arguments);
    CUdeviceptr* GetKernelArgument(KernelArgument& argument);
    size_t GetSharedMemorySize(const std::vector<KernelArgument*>& arguments) const;
    std::unique_ptr<CudaBuffer> CreateBuffer(KernelArgument& argument);
    std::unique_ptr<CudaBuffer> CreateUserBuffer(KernelArgument& argument, ComputeBuffer buffer);
    std::string GetDefaultCompilerOptions() const;
    void ClearStreamActions(const QueueId id);

#if defined(KTT_PROFILING_CUPTI)
    void InitializeCupti();
#endif // KTT_PROFILING_CUPTI

#if defined(KTT_PROFILING_CUPTI) || defined(KTT_PROFILING_CUPTI_LEGACY)
    void InitializeProfiling(const KernelComputeId& id);
    void FillProfilingData(const KernelComputeId& id, ComputationResult& result);
#endif // KTT_PROFILING_CUPTI || KTT_PROFILING_CUPTI_LEGACY
};

} // namespace ktt

#endif // KTT_API_CUDA
