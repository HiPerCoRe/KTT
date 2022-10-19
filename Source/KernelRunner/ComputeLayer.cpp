#include <Api/KttException.h>
#include <KernelRunner/ComputeLayer.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/Logger/Logger.h>
#include <Utility/Timer/Timer.h>
#include <Utility/StlHelpers.h>

namespace ktt
{

ComputeLayer::ComputeLayer(ComputeEngine& engine, KernelArgumentManager& argumentManager) :
    m_ComputeEngine(engine),
    m_ArgumentManager(argumentManager),
    m_ActiveKernel(InvalidKernelId)
{}

void ComputeLayer::RunKernel(const KernelDefinitionId id)
{
    const auto actionId = RunKernelAsync(id, GetDefaultQueue());
    WaitForComputeAction(actionId);
}

void ComputeLayer::RunKernel(const KernelDefinitionId id, const DimensionVector& globalSize,
    const DimensionVector& localSize)
{
    const auto actionId = RunKernelAsync(id, GetDefaultQueue(), globalSize, localSize);
    WaitForComputeAction(actionId);
}

ComputeActionId ComputeLayer::RunKernelAsync(const KernelDefinitionId id, const QueueId queue)
{
    const auto& data = GetComputeData(id);
    return m_ComputeEngine.RunKernelAsync(data, queue);
}

ComputeActionId ComputeLayer::RunKernelAsync(const KernelDefinitionId id, const QueueId queue, const DimensionVector& globalSize,
    const DimensionVector& localSize)
{
    auto data = GetComputeData(id);
    data.SetGlobalSize(globalSize);
    data.SetLocalSize(localSize);

    return m_ComputeEngine.RunKernelAsync(data, queue);
}

void ComputeLayer::WaitForComputeAction(const ComputeActionId id)
{
    const auto result = m_ComputeEngine.WaitForComputeAction(id);
    GetData().AddPartialResult(result);
}

void ComputeLayer::RunKernelWithProfiling(const KernelDefinitionId id)
{
    if (!GetData().IsProfilingEnabled(id))
    {
        throw KttException("Profiling is not enabled for kernel definition with id " + std::to_string(id));
    }

    const auto& data = GetComputeData(id);
    const auto result = m_ComputeEngine.RunKernelWithProfiling(data, GetDefaultQueue());
    GetData().AddPartialResult(result);
}

void ComputeLayer::RunKernelWithProfiling(const KernelDefinitionId id, const DimensionVector& globalSize,
    const DimensionVector& localSize)
{
    if (!GetData().IsProfilingEnabled(id))
    {
        throw KttException("Profiling is not enabled for kernel definition with id " + std::to_string(id));
    }

    auto data = GetComputeData(id);
    data.SetGlobalSize(globalSize);
    data.SetLocalSize(localSize);

    const auto result = m_ComputeEngine.RunKernelWithProfiling(data, GetDefaultQueue());
    GetData().AddPartialResult(result);
}

uint64_t ComputeLayer::GetRemainingProfilingRuns(const KernelDefinitionId id) const
{
    if (!GetData().IsProfilingEnabled(id))
    {
        throw KttException("Profiling is not enabled for kernel definition with id " + std::to_string(id));
    }

    const auto& data = GetComputeData(id);
    return m_ComputeEngine.GetRemainingProfilingRuns(data.GetUniqueIdentifier());
}

uint64_t ComputeLayer::GetRemainingProfilingRuns() const
{
    uint64_t result = 0;

    for (const auto* definition : GetData().GetKernel().GetDefinitions())
    {
        result += GetRemainingProfilingRuns(definition->GetId());
    }

    return result;
}

QueueId ComputeLayer::GetDefaultQueue() const
{
    return m_ComputeEngine.GetDefaultQueue();
}

std::vector<QueueId> ComputeLayer::GetAllQueues() const
{
    return m_ComputeEngine.GetAllQueues();
}

void ComputeLayer::SynchronizeQueue(const QueueId queue)
{
    m_ComputeEngine.SynchronizeQueue(queue);
}

void ComputeLayer::SynchronizeQueues()
{
    m_ComputeEngine.SynchronizeQueues();
}

void ComputeLayer::SynchronizeDevice()
{
    m_ComputeEngine.SynchronizeDevice();
}

const DimensionVector& ComputeLayer::GetCurrentGlobalSize(const KernelDefinitionId id) const
{
    return GetComputeData(id).GetGlobalSize();
}

const DimensionVector& ComputeLayer::GetCurrentLocalSize(const KernelDefinitionId id) const
{
    return GetComputeData(id).GetLocalSize();
}

const KernelConfiguration& ComputeLayer::GetCurrentConfiguration() const
{
    return GetData().GetConfiguration();
}

KernelRunMode ComputeLayer::GetRunMode() const
{
    return GetData().GetRunMode();
}

void ComputeLayer::ChangeArguments(const KernelDefinitionId id, const std::vector<ArgumentId>& arguments)
{
    if (!ContainsUniqueElements(arguments))
    {
        throw KttException("Kernel arguments for a single kernel definition must be unique");
    }

    auto newArguments = m_ArgumentManager.GetArguments(arguments);
    GetData().ChangeArguments(id, newArguments);
}

void ComputeLayer::SwapArguments(const KernelDefinitionId id, const ArgumentId& first, const ArgumentId& second)
{
    GetData().SwapArguments(id, first, second);
}

void ComputeLayer::UpdateScalarArgument(const ArgumentId& id, const void* data)
{
    const auto& argument = m_ArgumentManager.GetArgument(id);

    if (argument.GetMemoryType() != ArgumentMemoryType::Scalar)
    {
        throw KttException("Argument with id " + id + " is not a scalar argument");
    }

    auto argumentOverride = argument;
    argumentOverride.SetOwnedData(data, argument.GetDataSize());
    GetData().AddArgumentOverride(id, argumentOverride);
}

void ComputeLayer::UpdateLocalArgument(const ArgumentId& id, const size_t dataSize)
{
    const auto& argument = m_ArgumentManager.GetArgument(id);

    if (argument.GetMemoryType() != ArgumentMemoryType::Local)
    {
        throw KttException("Argument with id " + id + " is not a local memory argument");
    }

    auto argumentOverride = argument;
    argumentOverride.SetOwnedData(nullptr, dataSize);
    GetData().AddArgumentOverride(id, argumentOverride);
}

void ComputeLayer::UploadBuffer(const ArgumentId& id)
{
    Timer timer;
    timer.Start();

    const auto actionId = UploadBufferAsync(id, GetDefaultQueue());
    WaitForTransferAction(actionId);

    timer.Stop();
    GetData().IncreaseOverhead(timer.GetElapsedTime());
}

TransferActionId ComputeLayer::UploadBufferAsync(const ArgumentId& id, const QueueId queue)
{
    auto& argument = m_ArgumentManager.GetArgument(id);
    return m_ComputeEngine.UploadArgument(argument, queue);
}

void ComputeLayer::DownloadBuffer(const ArgumentId& id, void* destination, const size_t dataSize)
{
    Timer timer;
    timer.Start();

    const auto actionId = DownloadBufferAsync(id, GetDefaultQueue(), destination, dataSize);
    WaitForTransferAction(actionId);

    timer.Stop();
    GetData().IncreaseOverhead(timer.GetElapsedTime());
}

TransferActionId ComputeLayer::DownloadBufferAsync(const ArgumentId& id, const QueueId queue, void* destination,
    const size_t dataSize)
{
    return m_ComputeEngine.DownloadArgument(id, queue, destination, dataSize);
}

void ComputeLayer::UpdateBuffer(const ArgumentId& id, const void* data, const size_t dataSize)
{
    Timer timer;
    timer.Start();

    const auto actionId = UpdateBufferAsync(id, GetDefaultQueue(), data, dataSize);
    WaitForTransferAction(actionId);

    timer.Stop();
    GetData().IncreaseOverhead(timer.GetElapsedTime());
}

TransferActionId ComputeLayer::UpdateBufferAsync(const ArgumentId& id, const QueueId queue, const void* data,
    const size_t dataSize)
{
    return m_ComputeEngine.UpdateArgument(id, queue, data, dataSize);
}

void ComputeLayer::CopyBuffer(const ArgumentId& destination, const ArgumentId& source, const size_t dataSize)
{
    Timer timer;
    timer.Start();

    const auto actionId = CopyBufferAsync(destination, source, GetDefaultQueue(), dataSize);
    WaitForTransferAction(actionId);

    timer.Stop();
    GetData().IncreaseOverhead(timer.GetElapsedTime());
}

TransferActionId ComputeLayer::CopyBufferAsync(const ArgumentId& destination, const ArgumentId& source, const QueueId queue,
    const size_t dataSize)
{
    return m_ComputeEngine.CopyArgument(destination, queue, source, dataSize);
}

void ComputeLayer::WaitForTransferAction(const TransferActionId id)
{
    m_ComputeEngine.WaitForTransferAction(id);
}

void ComputeLayer::ResizeBuffer(const ArgumentId& id, const size_t newDataSize, const bool preserveData)
{
    Timer timer;
    timer.Start();

    m_ComputeEngine.ResizeArgument(id, newDataSize, preserveData);

    timer.Stop();
    GetData().IncreaseOverhead(timer.GetElapsedTime());
}

void ComputeLayer::ClearBuffer(const ArgumentId& id)
{
    Timer timer;
    timer.Start();

    m_ComputeEngine.ClearBuffer(id);

    timer.Stop();
    GetData().IncreaseOverhead(timer.GetElapsedTime());
}

bool ComputeLayer::HasBuffer(const ArgumentId& id)
{
    return m_ComputeEngine.HasBuffer(id);
}

void ComputeLayer::GetUnifiedMemoryBufferHandle(const ArgumentId& id, UnifiedBufferMemory& memoryHandle)
{
    m_ComputeEngine.GetUnifiedMemoryBufferHandle(id, memoryHandle);
}

void ComputeLayer::SetActiveKernel(const KernelId id)
{
    KttAssert(m_ActiveKernel == InvalidKernelId, "Unpaired call to set / clear active kernel");
    m_ActiveKernel = id;
}

void ComputeLayer::ClearActiveKernel()
{
    KttAssert(m_ActiveKernel != InvalidKernelId, "Unpaired call to set / clear active kernel");
    m_ActiveKernel = InvalidKernelId;
}

void ComputeLayer::ClearComputeEngineData(const KernelDefinitionId id)
{
    const auto computeId = GetData().GetComputeData(id).GetUniqueIdentifier();
    m_ComputeEngine.ClearData(computeId);
}

void ComputeLayer::ClearComputeEngineData()
{
    for (const auto* definition : GetData().GetKernel().GetDefinitions())
    {
        ClearComputeEngineData(definition->GetId());
    }
}

void ComputeLayer::AddData(const Kernel& kernel, const KernelConfiguration& configuration, const KernelDimensions& dimensions,
    const KernelRunMode mode)
{
    m_Data[kernel.GetId()] = std::make_unique<ComputeLayerData>(kernel, configuration, dimensions, mode);
}

void ComputeLayer::ClearData(const KernelId id)
{
    m_Data.erase(id);
}

KernelResult ComputeLayer::GenerateResult(const KernelId id, const Nanoseconds launcherDuration) const
{
    KttAssert(ContainsKey(m_Data, id), "Invalid compute layer data access");
    return m_Data.find(id)->second->GenerateResult(launcherDuration);
}

const ComputeLayerData& ComputeLayer::GetData() const
{
    KttAssert(m_ActiveKernel != InvalidKernelId, "Retrieving kernel compute data requires valid active kernel");
    return *m_Data.find(m_ActiveKernel)->second;
}

ComputeLayerData& ComputeLayer::GetData()
{
    return const_cast<ComputeLayerData&>(static_cast<const ComputeLayer*>(this)->GetData());
}

const KernelComputeData& ComputeLayer::GetComputeData(const KernelDefinitionId id) const
{
    return GetData().GetComputeData(id);
}

} // namespace ktt
