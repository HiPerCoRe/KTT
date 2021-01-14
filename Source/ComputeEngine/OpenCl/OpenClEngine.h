#pragma once

#ifdef KTT_API_OPENCL

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <Api/ComputeApiInitializer.h>
#include <ComputeEngine/OpenCl/Actions/OpenClComputeAction.h>
#include <ComputeEngine/OpenCl/Actions/OpenClTransferAction.h>
#include <ComputeEngine/OpenCl/OpenClBuffer.h>
#include <ComputeEngine/OpenCl/OpenClCommandQueue.h>
#include <ComputeEngine/OpenCl/OpenClContext.h>
#include <ComputeEngine/OpenCl/OpenClKernel.h>
#include <ComputeEngine/ActionIdGenerator.h>
#include <ComputeEngine/ComputeEngine.h>
#include <Utility/LruCache.h>

#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
#include <ComputeEngine/OpenCl/Gpa/GpaInterface.h>
#include <ComputeEngine/OpenCl/Gpa/GpaProfilingContext.h>
#include <ComputeEngine/OpenCl/Gpa/GpaProfilingInstance.h>
#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY

namespace ktt
{

class OpenClEngine : public ComputeEngine
{
public:
    explicit OpenClEngine(const PlatformIndex platformIndex, const DeviceIndex deviceIndex, const uint32_t queueCount);
    explicit OpenClEngine(const ComputeApiInitializer& initializer);

    // Kernel methods
    ComputeActionId RunKernelAsync(const KernelComputeData& data, const QueueId queue) override;
    KernelResult WaitForComputeAction(const ComputeActionId id) const override;

    // Profiling methods
    KernelResult RunKernelWithProfiling(const KernelComputeData& data, const QueueId queue) override;
    void SetProfilingCounters(const std::vector<std::string>& counters) override;
    bool IsProfilingSessionActive(const KernelComputeId& id) override;
    uint64_t GetRemainingProfilingRuns(const KernelComputeId& id) override;
    bool HasAccurateRemainingProfilingRuns() const override;

    // Buffer methods
    TransferActionId UploadArgumentAsync(const KernelArgument& kernelArgument, const QueueId queue) override;
    TransferActionId UpdateArgumentAsync(const ArgumentId id, const QueueId queue, const void* data,
        const size_t dataSize) override;
    TransferActionId DownloadArgumentAsync(const ArgumentId id, const QueueId queue, void* destination,
        const size_t dataSize) const override;
    TransferActionId CopyArgumentAsync(const ArgumentId destination, const QueueId queue, const ArgumentId source,
        const size_t dataSize) override;
    uint64_t WaitForTransferAction(const TransferActionId id) const override;
    void ResizeArgument(const ArgumentId id, const size_t newSize, const bool preserveData) override;
    void GetUnifiedMemoryBufferHandle(const ArgumentId id, UnifiedBufferMemory& handle) override;
    void AddCustomBuffer(const KernelArgument& kernelArgument, ComputeBuffer buffer) override;
    void ClearBuffer(const ArgumentId id) override;
    void ClearBuffers() override;

    // Queue methods
    QueueId GetDefaultQueue() const override;
    std::vector<QueueId> GetAllQueues() const override;
    void SynchronizeQueue(const QueueId queue) override;
    void SynchronizeDevice() override;

    // Information retrieval methods
    std::vector<PlatformInfo> GetPlatformInfo() const override;
    std::vector<DeviceInfo> GetDeviceInfo(const PlatformIndex platform) const override;
    DeviceInfo GetCurrentDeviceInfo() const override;

    // Utility methods
    void SetCompilerOptions(const std::string& options) override;
    void SetGlobalSizeType(const GlobalSizeType type) override;
    void SetAutomaticGlobalSizeCorrection(const bool flag) override;
    void SetKernelCacheCapacity(const uint64_t capacity) override;
    void ClearKernelCache() override;

private:
    PlatformIndex m_PlatformIndex;
    DeviceIndex m_DeviceIndex;
    std::string m_CompilerOptions;
    GlobalSizeType m_GlobalSizeType;
    ActionIdGenerator m_Generator;
    bool m_GlobalSizeCorrection;

    std::unique_ptr<OpenClContext> m_Context;
    std::vector<std::unique_ptr<OpenClCommandQueue>> m_Queues;
    std::map<ArgumentId, std::unique_ptr<OpenClBuffer>> m_Buffers;
    LruCache<KernelComputeId, std::shared_ptr<OpenClKernel>> m_KernelCache;
    std::map<ComputeActionId, std::unique_ptr<OpenClComputeAction>> m_ComputeActions;
    std::map<TransferActionId, std::unique_ptr<OpenClTransferAction>> m_TransferActions;

    //#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
    //std::unique_ptr<GpaInterface> m_GpaInterface;
    //std::unique_ptr<GpaProfilingContext> m_ProfilingContext;
    //std::map<KernelComputeId, std::vector<ComputeActionId>> m_KernelToActionMap;
    //std::map<KernelComputeId, std::unique_ptr<GpaProfilingInstance>> m_ProfilingInstances;
    //#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY

    //std::unique_ptr<OpenClProgram> CreateAndBuildProgram(const std::string& source) const;
    //void InitializeProfiler();
    //void SetKernelArgument(OpenClKernel& kernel, KernelArgument& argument);
    //void SetKernelArgument(OpenClKernel& kernel, KernelArgument& argument, const std::vector<LocalMemoryModifier>& modifiers);
    //ComputeActionId EnqueueKernel(OpenClKernel& kernel, const std::vector<size_t>& globalSize, const std::vector<size_t>& localSize,
    //    const QueueId queue, const uint64_t kernelLaunchOverhead) const;
    //KernelResult CreateKernelResult(const EventId id) const;
    //OpenClBuffer* FindBuffer(const ArgumentId id) const;
    //void SetKernelArgumentVector(OpenClKernel& kernel, const OpenClBuffer& buffer) const;
    //bool LoadBufferFromCache(const ArgumentId id, OpenClKernel& kernel) const;
    //void CheckLocalMemoryModifiers(const std::vector<KernelArgument*>& argumentPointers, const std::vector<LocalMemoryModifier>& modifiers) const;

    //#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
    //void InitializeProfiling(const KernelComputeId& id);
    //const KernelComputeId& GetKernelFromEvent(const EventId id) const;
    //static const std::vector<std::string>& GetDefaultGpaProfilingCounters();
    //void LaunchDummyPass(const KernelComputeId& id);
    //#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY
};

} // namespace ktt

#endif // KTT_API_OPENCL
