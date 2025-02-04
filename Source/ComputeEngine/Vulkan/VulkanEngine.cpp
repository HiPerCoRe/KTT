#ifdef KTT_API_VULKAN

#include <Api/KttException.h>
#include <ComputeEngine/Vulkan/VulkanEngine.h>
#include <ComputeEngine/Vulkan/VulkanUtility.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/Logger/Logger.h>
#include <Utility/Timer/Timer.h>
#include <Utility/StlHelpers.h>
#include <Utility/StringUtility.h>

namespace ktt
{

VulkanEngine::VulkanEngine(const DeviceIndex deviceIndex, const uint32_t queueCount) :
    m_Configuration(GlobalSizeType::Vulkan),
    m_DeviceIndex(deviceIndex),
    m_DeviceInfo(0, ""),
    m_PipelineCache(10)
{
    std::vector<const char*> extensions;
    std::vector<const char*> validationLayers;

#ifdef KTT_CONFIGURATION_DEBUG
    extensions.emplace_back("VK_EXT_debug_report");
    validationLayers.emplace_back("VK_LAYER_KHRONOS_validation");
#endif

    m_Instance = std::make_unique<VulkanInstance>("KTT Tuner", extensions, validationLayers);
    std::vector<VulkanPhysicalDevice> devices = m_Instance->GetPhysicalDevices();

    if (deviceIndex >= devices.size())
    {
        throw KttException("Invalid device index: " + std::to_string(deviceIndex));
    }

    m_Device = std::make_unique<VulkanDevice>(devices[deviceIndex], queueCount, VK_QUEUE_COMPUTE_BIT, std::vector<const char*>{},
        validationLayers);
    std::vector<VkQueue> queues = m_Device->GetQueues();

    for (size_t i = 0; i < queues.size(); ++i)
    {
        m_Queues.push_back(std::make_unique<VulkanQueue>(static_cast<QueueId>(i), queues[i]));
    }

    m_CommandPool = std::make_unique<VulkanCommandPool>(*m_Device, m_Device->GetQueueFamilyIndex());
    m_DescriptorPool = std::make_unique<VulkanDescriptorPool>(*m_Device, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        static_cast<uint32_t>(m_PipelineCache.GetMaxSize() * 10));
    m_QueryPool = std::make_unique<VulkanQueryPool>(*m_Device);
    m_Compiler = std::make_unique<ShadercCompiler>();
    m_Allocator = std::make_unique<VulkanMemoryAllocator>(*m_Instance, *m_Device);
    m_DeviceInfo = GetDeviceInfo(0)[m_DeviceIndex];
}

ComputeActionId VulkanEngine::RunKernelAsync(const KernelComputeData& data, const QueueId queueId, const bool powerMeasurementAllowed)
{
    if (queueId >= static_cast<QueueId>(m_Queues.size()))
    {
        throw KttException("Invalid queue index: " + std::to_string(queueId));
    }

    const uint64_t localSize = static_cast<uint64_t>(data.GetLocalSize().GetTotalSize());

    if (localSize > m_DeviceInfo.GetMaxWorkGroupSize())
    {
        throw KttException("Work-group size of " + std::to_string(localSize) + " exceeds current device limit",
            ExceptionReason::DeviceLimitsExceeded);
    }

    Timer timer;
    timer.Start();

    auto pipeline = LoadPipeline(data);
    std::vector<VulkanBuffer*> pipelineArguments = GetPipelineArguments(data.GetArguments());
    pipeline->BindArguments(pipelineArguments);
    
    std::vector<KernelArgument*> scalarArguments = GetScalarArguments(data.GetArguments());
    const auto& queue= *m_Queues[static_cast<size_t>(queueId)];

    timer.Stop();

    auto action = pipeline->DispatchShader(queue, *m_CommandPool, *m_QueryPool, data.GetGlobalSize(), scalarArguments);

    action->IncreaseOverhead(timer.GetElapsedTime());
    action->IncreaseCompilationOverhead(timer.GetElapsedTime()); //TODO check we are really measuring compilation time here
    action->SetComputeId(data.GetUniqueIdentifier());
    const auto id = action->GetId();
    m_ComputeActions[id] = std::move(action);
    return id;
}

ComputationResult VulkanEngine::WaitForComputeAction(const ComputeActionId id)
{
    if (!ContainsKey(m_ComputeActions, id))
    {
        throw KttException("Compute action with id " + std::to_string(id) + " was not found");
    }

    auto& action = *m_ComputeActions[id];
    action.WaitForFinish();
    auto result = action.GenerateResult();

    m_ComputeActions.erase(id);
    return result;
}

void VulkanEngine::ClearData(const KernelComputeId& id)
{
    EraseIf(m_ComputeActions, [&id](const auto& pair)
    {
        return pair.second->GetComputeId() == id;
    });
}

void VulkanEngine::ClearKernelData(const std::string& kernelName)
{
    EraseIf(m_ComputeActions, [&kernelName](const auto& pair)
    {
        return StartsWith(pair.second->GetComputeId(), kernelName);
    });
}

ComputationResult VulkanEngine::RunKernelWithProfiling([[maybe_unused]] const KernelComputeData& data,
    [[maybe_unused]] const QueueId queueId)
{
    throw KttException("Profiling is not yet supported for Vulkan backend");
}

void VulkanEngine::SetProfilingCounters([[maybe_unused]] const std::vector<std::string>& counters)
{
    throw KttException("Profiling is not yet supported for Vulkan backend");
}

bool VulkanEngine::IsProfilingSessionActive([[maybe_unused]] const KernelComputeId& id)
{
    throw KttException("Profiling is not yet supported for Vulkan backend");
}

uint64_t VulkanEngine::GetRemainingProfilingRuns([[maybe_unused]] const KernelComputeId& id)
{
    throw KttException("Profiling is not yet supported for Vulkan backend");
}

bool VulkanEngine::HasAccurateRemainingProfilingRuns() const
{
    return false;
}

bool VulkanEngine::SupportsMultiInstanceProfiling() const
{
    return false;
}

bool VulkanEngine::IsProfilingActive() const
{
    return false;
}

void VulkanEngine::SetProfiling(const bool profiling)
{
    throw KttException("Profiling is not yet supported for Vulkan backend");
}

TransferActionId VulkanEngine::UploadArgument(KernelArgument& kernelArgument, const QueueId queueId)
{
    Timer timer;
    timer.Start();

    const auto id = kernelArgument.GetId();
    Logger::LogDebug("Uploading buffer for argument with id " + id);

    if (queueId >= static_cast<QueueId>(m_Queues.size()))
    {
        throw KttException("Invalid queue index: " + std::to_string(queueId));
    }

    if (ContainsKey(m_Buffers, id))
    {
        throw KttException("Buffer for argument with id " + id + " already exists");
    }

    if (kernelArgument.GetMemoryType() != ArgumentMemoryType::Vector)
    {
        throw KttException("Argument with id " + id + " is not a vector and cannot be uploaded into buffer");
    }

    auto stagingBuffer = std::make_unique<VulkanBuffer>(kernelArgument, m_TransferIdGenerator, *m_Device, *m_Allocator,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
    auto stagingAction = stagingBuffer->UploadData(kernelArgument.GetData(), kernelArgument.GetDataSize());
    stagingAction->WaitForFinish();
    auto buffer = CreateBuffer(kernelArgument);

    timer.Stop();

    auto action = buffer->CopyData(*m_Queues[static_cast<size_t>(queueId)], *m_CommandPool, *m_QueryPool, std::move(stagingBuffer),
        kernelArgument.GetDataSize());
    action->IncreaseOverhead(timer.GetElapsedTime());
    const auto actionId = action->GetId();

    m_Buffers[id] = std::move(buffer);
    m_TransferActions[actionId] = std::move(action);

    return actionId;
}

TransferActionId VulkanEngine::UpdateArgument([[maybe_unused]] const ArgumentId& id, [[maybe_unused]] const QueueId queueId,
    [[maybe_unused]] const void* data, [[maybe_unused]] const size_t dataSize)
{
    throw KttException("Support for argument update is not yet available for Vulkan backend");
}

TransferActionId VulkanEngine::DownloadArgument(const ArgumentId& id, const QueueId queueId, void* destination,
    const size_t dataSize)
{
    Timer timer;
    timer.Start();

    Logger::LogDebug("Downloading buffer for argument with id " + id);

    if (queueId >= static_cast<QueueId>(m_Queues.size()))
    {
        throw KttException("Invalid queue index: " + std::to_string(queueId));
    }

    if (!ContainsKey(m_Buffers, id))
    {
        throw KttException("Buffer for argument with id " + id + " was not found");
    }

    auto& buffer = *m_Buffers[id];
    size_t actualDataSize = dataSize;

    if (actualDataSize == 0)
    {
        actualDataSize = buffer.GetSize();
    }

    auto stagingBuffer = std::make_unique<VulkanBuffer>(buffer.GetArgument(), m_TransferIdGenerator, *m_Device, *m_Allocator,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_ONLY);

    timer.Stop();

    // Todo: make this part asynchronous, handle data download inside action
    auto action = stagingBuffer->CopyData(*m_Queues[static_cast<size_t>(queueId)], *m_CommandPool, *m_QueryPool, buffer, actualDataSize);
    action->IncreaseOverhead(timer.GetElapsedTime());
    action->WaitForFinish();
    action->GenerateResult();

    auto downloadAction = stagingBuffer->DownloadData(destination, actualDataSize);

    const auto actionId = downloadAction->GetId();
    m_TransferActions[actionId] = std::move(downloadAction);
    return actionId;
}

TransferActionId VulkanEngine::CopyArgument([[maybe_unused]] const ArgumentId& destination, [[maybe_unused]] const QueueId queueId,
    [[maybe_unused]] const ArgumentId& source, [[maybe_unused]] const size_t dataSize)
{
    throw KttException("Support for argument copy is not yet available for Vulkan backend");
}

TransferResult VulkanEngine::WaitForTransferAction(const TransferActionId id)
{
    if (!ContainsKey(m_TransferActions, id))
    {
        throw KttException("Transfer action with id " + std::to_string(id) + " was not found");
    }

    auto& action = *m_TransferActions[id];
    action.WaitForFinish();
    auto result = action.GenerateResult();

    m_TransferActions.erase(id);
    return result;
}

void VulkanEngine::ResizeArgument([[maybe_unused]] const ArgumentId& id, [[maybe_unused]] const size_t newSize,
    [[maybe_unused]] const bool preserveData)
{
    throw KttException("Support for argument resize is not yet available for Vulkan backend");
}

void VulkanEngine::GetUnifiedMemoryBufferHandle([[maybe_unused]] const ArgumentId& id, [[maybe_unused]] UnifiedBufferMemory& handle)
{
    throw KttException("Support for unified memory buffers is not yet available for Vulkan backend");
}

void VulkanEngine::AddCustomBuffer([[maybe_unused]] KernelArgument& kernelArgument, [[maybe_unused]] ComputeBuffer buffer)
{
    throw KttException("Support for custom buffers is not yet available for Vulkan backend");
}

void VulkanEngine::ClearBuffer(const ArgumentId& id)
{
    m_Buffers.erase(id);
}

void VulkanEngine::ClearBuffers()
{
    m_Buffers.clear();
}

bool VulkanEngine::HasBuffer(const ArgumentId& id)
{
    return ContainsKey(m_Buffers, id);
}

QueueId VulkanEngine::AddComputeQueue([[maybe_unused]] ComputeQueue queue)
{
    throw KttException("Support for compute queue addition is not yet available for Vulkan backend");
}

void VulkanEngine::RemoveComputeQueue([[maybe_unused]] const QueueId id)
{
    throw KttException("Support for compute queue removal is not yet available for Vulkan backend");
}

QueueId VulkanEngine::GetDefaultQueue() const
{
    return static_cast<QueueId>(0);
}

std::vector<QueueId> VulkanEngine::GetAllQueues() const
{
    std::vector<QueueId> result;

    for (const auto& queue : m_Queues)
    {
        result.push_back(queue->GetId());
    }

    return result;
}

void VulkanEngine::SynchronizeQueue(const QueueId queueId)
{
    if (static_cast<size_t>(queueId) >= m_Queues.size())
    {
        throw KttException("Invalid Vulkan queue index: " + std::to_string(queueId));
    }

    m_Queues[static_cast<size_t>(queueId)]->WaitIdle();
    ClearQueueActions(queueId);
}

void VulkanEngine::SynchronizeQueues()
{
    for (auto& queue : m_Queues)
    {
        queue->WaitIdle();
        ClearQueueActions(queue->GetId());
    }
}

void VulkanEngine::SynchronizeDevice()
{
    m_Device->WaitIdle();
    m_ComputeActions.clear();
    m_TransferActions.clear();
}

std::vector<PlatformInfo> VulkanEngine::GetPlatformInfo() const
{
    PlatformInfo info(0, "Vulkan");
    info.SetVendor("N/A");
    info.SetVersion(m_Instance->GetApiVersion());

    std::vector<std::string> extensions = m_Instance->GetExtensions();
    std::string mergedExtensions;

    for (size_t i = 0; i < extensions.size(); ++i)
    {
        mergedExtensions += extensions[i];

        if (i != extensions.size() - 1)
        {
            mergedExtensions += ", ";
        }
    }

    info.SetExtensions(mergedExtensions);
    return std::vector<PlatformInfo>{info};
}

std::vector<DeviceInfo> VulkanEngine::GetDeviceInfo([[maybe_unused]] const PlatformIndex platformIndex) const
{
    std::vector<DeviceInfo> result;
    std::vector<VulkanPhysicalDevice> devices = m_Instance->GetPhysicalDevices();

    for (const auto& device : devices)
    {
        result.push_back(device.GetInfo());
    }

    return result;
}

PlatformInfo VulkanEngine::GetCurrentPlatformInfo() const
{
    return GetPlatformInfo()[0];
}

DeviceInfo VulkanEngine::GetCurrentDeviceInfo() const
{
    return m_DeviceInfo;
}

ComputeApi VulkanEngine::GetComputeApi() const
{
    return ComputeApi::Vulkan;
}

GlobalSizeType VulkanEngine::GetGlobalSizeType() const
{
    return m_Configuration.GetGlobalSizeType();
}

void VulkanEngine::SetCompilerOptions(const std::string& options, [[maybe_unused]] const bool overrideDefault)
{
    m_Configuration.SetCompilerOptions(options);
    ClearKernelCache();
}

void VulkanEngine::SetGlobalSizeType(const GlobalSizeType type)
{
    m_Configuration.SetGlobalSizeType(type);
}

void VulkanEngine::SetAutomaticGlobalSizeCorrection(const bool flag)
{
    m_Configuration.SetGlobalSizeCorrection(flag);
}

void VulkanEngine::SetKernelCacheCapacity(const uint64_t capacity)
{
    m_PipelineCache.SetMaxSize(static_cast<size_t>(capacity));
}

void VulkanEngine::ClearKernelCache()
{
    m_PipelineCache.Clear();
}

void VulkanEngine::EnsureThreadContext()
{}

std::shared_ptr<VulkanComputePipeline> VulkanEngine::LoadPipeline(const KernelComputeData& data)
{
    const auto id = data.GetUniqueIdentifier();

    if (m_PipelineCache.GetMaxSize() > 0 && m_PipelineCache.Exists(id))
    {
        return m_PipelineCache.Get(id)->second;
    }

    auto pipeline = std::make_shared<VulkanComputePipeline>(*m_Device, m_ComputeIdGenerator, *m_Compiler, data.GetName(),
        data.GetDefaultSource(), data.GetLocalSize(), data.GetConfiguration(), *m_DescriptorPool, data.GetArguments());

    if (m_PipelineCache.GetMaxSize() > 0)
    {
        m_PipelineCache.Put(id, pipeline);
    }

    return pipeline;
}

VulkanBuffer* VulkanEngine::GetPipelineArgument(KernelArgument& argument)
{
    switch (argument.GetMemoryType())
    {
    case ArgumentMemoryType::Scalar:
    case ArgumentMemoryType::Local:
    case ArgumentMemoryType::Symbol:
        KttError("Scalar, symbol and local memory arguments do not have Vulkan buffer representation");
        return nullptr;
    case ArgumentMemoryType::Vector:
    {
        const auto id = argument.GetId();

        if (!ContainsKey(m_Buffers, id))
        {
            throw KttException("Buffer corresponding to kernel argument with id " + id + " was not found");
        }

        return m_Buffers[id].get();
    }
    default:
        KttError("Unhandled argument memory type value");
        return nullptr;
    }
}

std::vector<VulkanBuffer*> VulkanEngine::GetPipelineArguments(const std::vector<KernelArgument*>& arguments)
{
    std::vector<VulkanBuffer*> result;

    for (auto* argument : arguments)
    {
        if (argument->GetMemoryType() != ArgumentMemoryType::Vector)
        {
            continue;
        }

        VulkanBuffer* buffer = GetPipelineArgument(*argument);
        result.push_back(buffer);
    }

    return result;
}

std::unique_ptr<VulkanBuffer> VulkanEngine::CreateBuffer(KernelArgument& argument)
{
    std::unique_ptr<VulkanBuffer> buffer;

    switch (argument.GetMemoryLocation())
    {
    case ArgumentMemoryLocation::Undefined:
        KttError("Buffer cannot be created for arguments with undefined memory location");
        break;
    case ArgumentMemoryLocation::Device:
        buffer = std::make_unique<VulkanBuffer>(argument, m_TransferIdGenerator, *m_Device, *m_Allocator,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
        break;
    case ArgumentMemoryLocation::Host:
    case ArgumentMemoryLocation::HostZeroCopy:
        throw KttException("Support for host buffers is not yet available for Vulkan backend");
        break;
    case ArgumentMemoryLocation::Unified:
        throw KttException("Support for unified buffers is not yet available for Vulkan backend");
        break;
    default:
        KttError("Unhandled argument memory location value");
        break;
    }

    return buffer;
}

std::unique_ptr<VulkanBuffer> VulkanEngine::CreateUserBuffer([[maybe_unused]] KernelArgument& argument,
    [[maybe_unused]] ComputeBuffer buffer)
{
    throw KttException("Support for custom buffers is not yet available for Vulkan backend");
}

void VulkanEngine::ClearQueueActions(const QueueId id)
{
    EraseIf(m_ComputeActions, [id](const auto& pair)
    {
        return pair.second->GetQueueId() == id || pair.second->GetQueueId() == InvalidQueueId;
    });

    EraseIf(m_TransferActions, [id](const auto& pair)
    {
        return pair.second->GetQueueId() == id || pair.second->GetQueueId() == InvalidQueueId;
    });
}

std::vector<KernelArgument*> VulkanEngine::GetScalarArguments(const std::vector<KernelArgument*>& arguments)
{
    std::vector<KernelArgument*> result;

    for (auto* argument : arguments)
    {
        if (argument->GetMemoryType() == ArgumentMemoryType::Scalar || argument->GetMemoryType() == ArgumentMemoryType::Symbol)
        {
            result.push_back(argument);
        }
    }

    return result;
}

} // namespace ktt

#endif // KTT_API_VULKAN
