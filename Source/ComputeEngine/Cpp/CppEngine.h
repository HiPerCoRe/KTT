#pragma once

#ifdef KTT_API_CPP

#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <vector>
#include <thread>
#include <future>
#include <functional>

#include <Api/ComputeApiInitializer.h>
#include <Api/Configuration/PowerMeasurementParameters.h>
#include <ComputeEngine/ComputeEngine.h>
#include <ComputeEngine/Cpp/CppCompiler.h>
#include <ComputeEngine/EngineConfiguration.h>
#include <Utility/IdGenerator.h>
#include <Utility/LruCache.h>

namespace ktt
{

class CppEngine : public ComputeEngine
{
public:
    explicit CppEngine(const PlatformIndex platformIndex, const DeviceIndex deviceIndex, const uint32_t queueCount);
    explicit CppEngine(const ComputeApiInitializer& initializer, std::vector<QueueId>& assignedQueueIds);

    // Kernel methods
    ComputeActionId RunKernelAsync(const KernelComputeData& data, const QueueId queueId, const bool powerMeasurementAllowed = false,
        const std::optional<PowerMeasurementParameters>& powerParams = std::nullopt) override;
    ComputationResult WaitForComputeAction(const ComputeActionId id) override;
    void ClearData(const KernelComputeId& id) override;
    void ClearKernelData(const std::string& kernelName) override;

    // Profiling methods
    ComputationResult RunKernelWithProfiling(const KernelComputeData& data, const QueueId queueId,
        const std::optional<PowerMeasurementParameters>& powerParams = std::nullopt) override;
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
    void SetCompiler(const std::string& compiler) override;

private:
    struct CppKernel
    {
        CppCompiler::KernelFunction function;
        std::vector<void*> argumentPointers;
        std::vector<size_t> argumentSizes;
        std::vector<size_t> scalarValues; // Storage for scalar argument values
        Nanoseconds compilationOverhead = 0; // Time spent compiling this kernel
    };

    struct CppBuffer
    {
        void* data = nullptr;
        size_t size = 0;
        ArgumentMemoryLocation location;
        ArgumentAccessType access;
        bool owned = true; // Whether this buffer owns the memory (should free it)

        // Default constructor
        CppBuffer() = default;

        // Destructor - frees allocated memory only if owned
        ~CppBuffer()
        {
            if (owned && data != nullptr)
            {
                free(data);
                data = nullptr;
            }
        }

        // Copy constructor - performs deep copy of data (new copy is always owned)
        CppBuffer(const CppBuffer& other)
            : size(other.size)
            , location(other.location)
            , access(other.access)
            , owned(true)
        {
            if (other.data != nullptr && size > 0)
            {
                data = malloc(size);
                memcpy(data, other.data, size);
            }
        }

        // Copy assignment - performs deep copy of data (result is always owned)
        CppBuffer& operator=(const CppBuffer& other)
        {
            if (this != &other)
            {
                // Free existing data only if owned
                if (owned && data != nullptr)
                {
                    free(data);
                    data = nullptr;
                }

                size = other.size;
                location = other.location;
                access = other.access;
                owned = true; // New copy is owned

                if (other.data != nullptr && size > 0)
                {
                    data = malloc(size);
                    memcpy(data, other.data, size);
                }
            }
            return *this;
        }

        // Move constructor
        CppBuffer(CppBuffer&& other) noexcept
            : data(other.data)
            , size(other.size)
            , location(other.location)
            , access(other.access)
            , owned(other.owned)
        {
            other.data = nullptr;
            other.size = 0;
            other.owned = true;
        }

        // Move assignment
        CppBuffer& operator=(CppBuffer&& other) noexcept
        {
            if (this != &other)
            {
                // Free existing data only if owned
                if (owned && data != nullptr)
                {
                    free(data);
                }

                data = other.data;
                size = other.size;
                location = other.location;
                access = other.access;
                owned = other.owned;

                other.data = nullptr;
                other.size = 0;
                other.owned = true;
            }
            return *this;
        }
    };

    struct QueueData
    {
        std::thread thread;
        std::vector<std::future<ComputationResult>> pendingActions;
    };

    EngineConfiguration m_Configuration;
    PlatformIndex m_PlatformIndex;
    DeviceIndex m_DeviceIndex;
    DeviceInfo m_DeviceInfo;
    IdGenerator<QueueId> m_QueueIdGenerator;
    IdGenerator<ComputeActionId> m_ComputeIdGenerator;
    IdGenerator<TransferActionId> m_TransferIdGenerator;
    std::map<QueueId, QueueData> m_Queues;
    std::map<ArgumentId, CppBuffer> m_Buffers;
    LruCache<KernelComputeId, std::shared_ptr<CppKernel>> m_KernelCache;
    std::map<ComputeActionId, std::future<ComputationResult>> m_ComputeActions;
    std::map<ComputeActionId, KernelComputeId> m_ComputeActionToKernel;
    std::map<TransferActionId, std::future<TransferResult>> m_TransferActions;
    CppCompiler m_Compiler;

    std::shared_ptr<CppKernel> LoadKernel(const KernelComputeData& data);
    void SetKernelArguments(CppKernel& kernel, const std::vector<KernelArgument*> arguments);
    void SetKernelArgument(CppKernel& kernel, const KernelArgument& argument);
    std::unique_ptr<CppBuffer> CreateBuffer(KernelArgument& argument);
    std::unique_ptr<CppBuffer> CreateUserBuffer(KernelArgument& argument, ComputeBuffer buffer);
    void ClearQueueActions(const QueueId id);
    std::string GetDefaultCompilerOptions() const;
};

} // namespace ktt

#endif // KTT_API_CPP
