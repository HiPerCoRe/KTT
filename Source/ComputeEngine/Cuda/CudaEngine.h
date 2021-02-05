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
#include <Utility/IdGenerator.h>
#include <Utility/LruCache.h>

#ifdef KTT_PROFILING_CUPTI_LEGACY
#include <ComputeEngine/Cuda/CuptiLegacy/CuptiInstance.h>
#elif KTT_PROFILING_CUPTI
#include <ComputeEngine/Cuda/Cupti/CuptiInstance.h>
#include <ComputeEngine/Cuda/Cupti/CuptiMetricInterface.h>
#include <ComputeEngine/Cuda/Cupti/CuptiProfiler.h>
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
    ComputationResult WaitForComputeAction(const ComputeActionId id) override;

    // Profiling methods
    ComputationResult RunKernelWithProfiling(const KernelComputeData& data, const QueueId queueId) override;
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
    IdGenerator<ComputeActionId> m_ComputeIdGenerator;
    IdGenerator<TransferActionId> m_TransferIdGenerator;
    std::unique_ptr<CudaContext> m_Context;
    std::vector<std::unique_ptr<CudaStream>> m_Streams;
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

    std::shared_ptr<CudaKernel> LoadKernel(const KernelComputeData& data);
    std::vector<CUdeviceptr*> GetKernelArguments(const std::vector<KernelArgument*>& arguments);
    CUdeviceptr* GetKernelArgument(KernelArgument& argument);
    size_t GetSharedMemorySize(const std::vector<KernelArgument*>& arguments) const;
    std::unique_ptr<CudaBuffer> CreateBuffer(KernelArgument& argument);
    std::unique_ptr<CudaBuffer> CreateUserBuffer(KernelArgument& argument, ComputeBuffer buffer);

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
