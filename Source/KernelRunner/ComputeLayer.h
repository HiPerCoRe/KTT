#pragma once

#include <map>
#include <memory>

#include <Api/Output/KernelResult.h>
#include <Api/ComputeInterface.h>
#include <ComputeEngine/ComputeEngine.h>
#include <KernelArgument/KernelArgumentManager.h>
#include <KernelRunner/ComputeLayerData.h>
#include <KttTypes.h>

namespace ktt
{

class Kernel;
class KernelConfiguration;

class ComputeLayer : public ComputeInterface
{
public:
    explicit ComputeLayer(ComputeEngine& engine, KernelArgumentManager& argumentManager);

    void RunKernel(const KernelDefinitionId id) override;
    void RunKernel(const KernelDefinitionId id, const DimensionVector& globalSize, const DimensionVector& localSize) override;
    ComputeActionId RunKernelAsync(const KernelDefinitionId id, const QueueId queue) override;
    ComputeActionId RunKernelAsync(const KernelDefinitionId id, const QueueId queue, const DimensionVector& globalSize,
        const DimensionVector& localSize) override;
    void WaitForComputeAction(const ComputeActionId id) override;
    void ClearKernelData(const KernelDefinitionId id) override;
    void ClearKernelData() override;

    void RunKernelWithProfiling(const KernelDefinitionId id) override;
    void RunKernelWithProfiling(const KernelDefinitionId id, const DimensionVector& globalSize,
        const DimensionVector& localSize) override;
    uint64_t GetRemainingProfilingRuns(const KernelDefinitionId id) const override;
    uint64_t GetRemainingProfilingRuns() const override;

    QueueId GetDefaultQueue() const override;
    std::vector<QueueId> GetAllQueues() const override;
    void SynchronizeQueue(const QueueId queue) override;
    void SynchronizeDevice() override;

    const DimensionVector& GetCurrentGlobalSize(const KernelDefinitionId id) const override;
    const DimensionVector& GetCurrentLocalSize(const KernelDefinitionId id) const override;
    const KernelConfiguration& GetCurrentConfiguration() const override;

    void ChangeArguments(const KernelDefinitionId id, const std::vector<ArgumentId>& arguments) override;
    void SwapArguments(const KernelDefinitionId id, const ArgumentId first, const ArgumentId second) override;
    void UpdateScalarArgument(const ArgumentId id, const void* data) override;
    void UpdateLocalArgument(const ArgumentId id, const size_t dataSize) override;

    TransferActionId UploadBuffer(const ArgumentId id, const QueueId queue) override;
    TransferActionId DownloadBuffer(const ArgumentId id, const QueueId queue, void* destination, const size_t dataSize) override;
    TransferActionId UpdateBuffer(const ArgumentId id, const QueueId queue, const void* data, const size_t dataSize) override;
    TransferActionId CopyBuffer(const ArgumentId destination, const ArgumentId source, const QueueId queue,
        const size_t dataSize) override;
    void WaitForTransferAction(const TransferActionId id) override;
    void ResizeBuffer(const ArgumentId id, const size_t newDataSize, const bool preserveData) override;
    void ClearBuffer(const ArgumentId id) override;
    bool HasBuffer(const ArgumentId id) override;
    void GetUnifiedMemoryBufferHandle(const ArgumentId id, UnifiedBufferMemory& memoryHandle) override;

    void SetActiveKernel(const KernelId id);
    void ClearActiveKernel();

    void AddData(const Kernel& kernel, const KernelConfiguration& configuration);
    void ClearData(const KernelId id);
    KernelResult GenerateResult(const KernelId id, const Nanoseconds launcherDuration) const;

private:
    std::map<KernelId, std::unique_ptr<ComputeLayerData>> m_Data;
    ComputeEngine& m_ComputeEngine;
    KernelArgumentManager& m_ArgumentManager;
    KernelId m_ActiveKernel;

    const ComputeLayerData& GetData() const;
    ComputeLayerData& GetData();
    const KernelComputeData& GetComputeData(const KernelDefinitionId id) const;
};

} // namespace ktt
