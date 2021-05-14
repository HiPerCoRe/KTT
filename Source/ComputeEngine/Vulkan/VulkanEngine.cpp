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
    m_DeviceIndex(deviceIndex),
    m_DeviceInfo(0, ""),
    m_CompilerOptions(""),
    m_GlobalSizeType(GlobalSizeType::Vulkan),
    m_GlobalSizeCorrection(false),
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

/*ComputeActionId VulkanEngine::RunKernelAsync(const KernelComputeData& data, const QueueId queueId)
{
    Timer overheadTimer;
    overheadTimer.Start();

    VulkanComputePipeline* pipeline;
    std::unique_ptr<VulkanComputePipeline> pipelineUnique;
    std::unique_ptr<VulkanDescriptorSetLayout> layout;
    std::unique_ptr<VulkanShaderModule> shader;

    std::vector<VulkanBuffer*> pipelineArguments = getPipelineArguments(argumentPointers);
    const uint32_t bindingCount = static_cast<uint32_t>(pipelineArguments.size());
    std::vector<KernelArgument*> scalarArguments = getScalarArguments(argumentPointers);
    VulkanPushConstant pushConstant(scalarArguments);

    if (kernelCacheFlag)
    {
        if (pipelineCache.find(std::make_pair(kernelData.getName(), kernelData.getSource())) == pipelineCache.end())
        {
            if (pipelineCache.size() >= kernelCacheCapacity)
            {
                clearKernelCache();
            }
            auto cacheLayout = std::make_unique<VulkanDescriptorSetLayout>(device->getDevice(), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, bindingCount);
            auto cacheShader = std::make_unique<VulkanShaderModule>(device->getDevice(), kernelData.getName(), kernelData.getUnmodifiedSource(),
                kernelData.getLocalSize(), kernelData.getParameterPairs());
            auto cachePipeline = std::make_unique<VulkanComputePipeline>(device->getDevice(), cacheLayout->getDescriptorSetLayout(),
                cacheShader->getShaderModule(), kernelData.getName(), pushConstant);
            auto cacheEntry = std::make_unique<VulkanPipelineCacheEntry>(std::move(cachePipeline), std::move(cacheLayout), std::move(cacheShader));
            pipelineCache.insert(std::make_pair(std::make_pair(kernelData.getName(), kernelData.getSource()), std::move(cacheEntry)));
        }
        auto cachePointer = pipelineCache.find(std::make_pair(kernelData.getName(), kernelData.getSource()));
        pipeline = cachePointer->second->pipeline.get();
    }
    else
    {
        layout = std::make_unique<VulkanDescriptorSetLayout>(device->getDevice(), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, bindingCount);
        shader = std::make_unique<VulkanShaderModule>(device->getDevice(), kernelData.getName(), kernelData.getUnmodifiedSource(),
            kernelData.getLocalSize(), kernelData.getParameterPairs());
        pipelineUnique = std::make_unique<VulkanComputePipeline>(device->getDevice(), layout->getDescriptorSetLayout(), shader->getShaderModule(),
            kernelData.getName(), pushConstant);
        pipeline = pipelineUnique.get();
    }

    pipeline->bindArguments(pipelineArguments);
    overheadTimer.stop();

    return enqueuePipeline(*pipeline, kernelData.getGlobalSize(), kernelData.getLocalSize(), queue, overheadTimer.getElapsedTime(), pushConstant);
}

KernelResult VulkanEngine::getKernelResult(const EventId id, const std::vector<OutputDescriptor>& outputDescriptors) const
{
    KernelResult result = createKernelResult(id);

    for (const auto& descriptor : outputDescriptors)
    {
        downloadArgument(descriptor.getArgumentId(), descriptor.getOutputDestination(), descriptor.getOutputSizeInBytes());
    }

    return result;
}

uint64_t VulkanEngine::getKernelOverhead(const EventId id) const
{
    auto eventPointer = kernelEvents.find(id);

    if (eventPointer == kernelEvents.end())
    {
        throw std::runtime_error(std::string("Kernel event with following id does not exist or its result was already retrieved: ")
            + std::to_string(id));
    }

    return eventPointer->second->getOverhead();
}

void VulkanEngine::clearEvents()
{
    kernelEvents.clear();
    bufferEvents.clear();
    eventCommands.clear();
    stagingBuffers.clear();
}

uint64_t VulkanEngine::uploadArgument(KernelArgument& kernelArgument)
{
    if (kernelArgument.getUploadType() != ArgumentUploadType::Vector)
    {
        return 0;
    }

    EventId eventId = uploadArgumentAsync(kernelArgument, getDefaultQueue());
    return getArgumentOperationDuration(eventId);
}

EventId VulkanEngine::uploadArgumentAsync(KernelArgument& kernelArgument, const QueueId queue)
{
    if (kernelArgument.getMemoryLocation() == ArgumentMemoryLocation::HostZeroCopy)
    {
        throw std::runtime_error("Host zero-copy arguments are not supported yet for Vulkan backend");
    }

    if (queue >= queues.size())
    {
        throw std::runtime_error(std::string("Invalid queue index: ") + std::to_string(queue));
    }

    if (findBuffer(kernelArgument.getId()) != nullptr)
    {
        throw std::runtime_error(std::string("Buffer with following id already exists: ") + std::to_string(kernelArgument.getId()));
    }

    if (kernelArgument.getUploadType() != ArgumentUploadType::Vector)
    {
        return 0;
    }

    EventId eventId = nextEventId;
    Logger::logDebug("Uploading buffer for argument " + std::to_string(kernelArgument.getId()) + ", event id: " + std::to_string(eventId));

    VkBufferUsageFlags hostUsage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (kernelArgument.getMemoryLocation() == ArgumentMemoryLocation::Host)
    {
        hostUsage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        auto bufferEvent = std::make_unique<VulkanEvent>(device->getDevice(), eventId, false);
        bufferEvents.insert(std::make_pair(eventId, std::move(bufferEvent)));
    }

    auto hostBuffer = std::make_unique<VulkanBuffer>(kernelArgument, device->getDevice(), device->getPhysicalDevice(), hostUsage);
    hostBuffer->allocateMemory(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    hostBuffer->uploadData(kernelArgument.getData(), kernelArgument.getDataSizeInBytes());

    if (kernelArgument.getMemoryLocation() == ArgumentMemoryLocation::Device)
    {
        VkBufferUsageFlags deviceUsage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (kernelArgument.getAccessType() != ArgumentAccessType::ReadOnly)
        {
            deviceUsage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        }

        auto deviceBuffer = std::make_unique<VulkanBuffer>(kernelArgument, device->getDevice(), device->getPhysicalDevice(), deviceUsage);
        deviceBuffer->allocateMemory(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        auto bufferEvent = std::make_unique<VulkanEvent>(device->getDevice(), eventId, true);
        auto commandBuffer = std::make_unique<VulkanCommandBufferHolder>(device->getDevice(), commandPool->getCommandPool());
        deviceBuffer->recordCopyDataCommand(commandBuffer->getCommandBuffer(), hostBuffer->getBuffer(), hostBuffer->getBufferSize());
        queues[queue].submitSingleCommand(commandBuffer->getCommandBuffer(), bufferEvent->getFence().getFence());

        bufferEvents.insert(std::make_pair(eventId, std::move(bufferEvent)));
        eventCommands.insert(std::make_pair(eventId, std::move(commandBuffer)));
        stagingBuffers.insert(std::make_pair(eventId, std::move(hostBuffer)));
        buffers.insert(std::move(deviceBuffer));
    }
    else if (kernelArgument.getMemoryLocation() == ArgumentMemoryLocation::Host)
    {
        buffers.insert(std::move(hostBuffer));
    }

    ++nextEventId;
    return eventId;
}

EventId VulkanEngine::updateArgumentAsync(const ArgumentId id, const void* data, const size_t dataSizeInBytes, const QueueId queue)
{
    throw std::runtime_error("Vulkan API is not yet supported");
}

EventId VulkanEngine::downloadArgumentAsync(const ArgumentId id, void* destination, const size_t dataSizeInBytes, const QueueId queue) const
{
    if (queue >= queues.size())
    {
        throw std::runtime_error(std::string("Invalid queue index: ") + std::to_string(queue));
    }

    VulkanBuffer* buffer = findBuffer(id);

    if (buffer == nullptr)
    {
        throw std::runtime_error(std::string("Buffer with following id was not found: ") + std::to_string(id));
    }

    EventId eventId = nextEventId;
    Logger::logDebug("Downloading buffer for argument " + std::to_string(id) + ", event id: " + std::to_string(eventId));
    size_t actualDataSize = buffer->getBufferSize();
    if (dataSizeInBytes > 0)
    {
        actualDataSize = dataSizeInBytes;
    }

    if (buffer->getMemoryLocation() == ArgumentMemoryLocation::Host)
    {
        buffer->downloadData(destination, actualDataSize);
        auto bufferEvent = std::make_unique<VulkanEvent>(device->getDevice(), eventId, false);
        bufferEvents.insert(std::make_pair(eventId, std::move(bufferEvent)));
    }
    else if (buffer->getMemoryLocation() == ArgumentMemoryLocation::Device)
    {
        auto hostBuffer = std::make_unique<VulkanBuffer>(*buffer, device->getDevice(), device->getPhysicalDevice(), VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            actualDataSize);
        hostBuffer->allocateMemory(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        auto bufferEvent = std::make_unique<VulkanEvent>(device->getDevice(), eventId, true);
        auto commandBuffer = std::make_unique<VulkanCommandBufferHolder>(device->getDevice(), commandPool->getCommandPool());
        hostBuffer->recordCopyDataCommand(commandBuffer->getCommandBuffer(), buffer->getBuffer(), actualDataSize);
        queues[queue].submitSingleCommand(commandBuffer->getCommandBuffer(), bufferEvent->getFence().getFence());

        // todo: make this asynchronous
        bufferEvent->wait();
        hostBuffer->downloadData(destination, actualDataSize);

        bufferEvents.insert(std::make_pair(eventId, std::move(bufferEvent)));
        eventCommands.insert(std::make_pair(eventId, std::move(commandBuffer)));
        stagingBuffers.insert(std::make_pair(eventId, std::move(hostBuffer)));
    }

    nextEventId++;
    return eventId;
}

KernelArgument VulkanEngine::downloadArgumentObject(const ArgumentId id, uint64_t* downloadDuration) const
{
    VulkanBuffer* buffer = findBuffer(id);

    if (buffer == nullptr)
    {
        throw std::runtime_error(std::string("Buffer with following id was not found: ") + std::to_string(id));
    }

    KernelArgument argument(buffer->getKernelArgumentId(), buffer->getBufferSize() / buffer->getElementSize(), buffer->getElementSize(),
        buffer->getDataType(), buffer->getMemoryLocation(), buffer->getAccessType(), ArgumentUploadType::Vector);

    EventId eventId = nextEventId;
    Logger::logDebug("Downloading buffer for argument " + std::to_string(id) + ", event id: " + std::to_string(eventId));

    if (buffer->getMemoryLocation() == ArgumentMemoryLocation::Host)
    {
        buffer->downloadData(argument.getData(), argument.getDataSizeInBytes());
        auto bufferEvent = std::make_unique<VulkanEvent>(device->getDevice(), eventId, false);
        bufferEvents.insert(std::make_pair(eventId, std::move(bufferEvent)));
    }
    else if (buffer->getMemoryLocation() == ArgumentMemoryLocation::Device)
    {
        auto hostBuffer = std::make_unique<VulkanBuffer>(*buffer, device->getDevice(), device->getPhysicalDevice(), VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            argument.getDataSizeInBytes());
        hostBuffer->allocateMemory(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        auto bufferEvent = std::make_unique<VulkanEvent>(device->getDevice(), eventId, true);
        auto commandBuffer = std::make_unique<VulkanCommandBufferHolder>(device->getDevice(), commandPool->getCommandPool());
        hostBuffer->recordCopyDataCommand(commandBuffer->getCommandBuffer(), buffer->getBuffer(), argument.getDataSizeInBytes());
        queues[getDefaultQueue()].submitSingleCommand(commandBuffer->getCommandBuffer(), bufferEvent->getFence().getFence());

        bufferEvent->wait();
        hostBuffer->downloadData(argument.getData(), argument.getDataSizeInBytes());

        bufferEvents.insert(std::make_pair(eventId, std::move(bufferEvent)));
        eventCommands.insert(std::make_pair(eventId, std::move(commandBuffer)));
        stagingBuffers.insert(std::make_pair(eventId, std::move(hostBuffer)));
    }

    nextEventId++;

    uint64_t duration = getArgumentOperationDuration(eventId);
    if (downloadDuration != nullptr)
    {
        *downloadDuration = duration;
    }

    return argument;
}

uint64_t VulkanEngine::copyArgument(const ArgumentId destination, const ArgumentId source, const size_t dataSizeInBytes)
{
    EventId eventId = copyArgumentAsync(destination, source, dataSizeInBytes, getDefaultQueue());
    return getArgumentOperationDuration(eventId);
}

EventId VulkanEngine::copyArgumentAsync(const ArgumentId destination, const ArgumentId source, const size_t dataSizeInBytes, const QueueId queue)
{
    throw std::runtime_error("Vulkan API is not yet supported");
}

uint64_t VulkanEngine::persistArgument(KernelArgument& kernelArgument, const bool flag)
{
    throw std::runtime_error("Vulkan API is not yet supported");
}

uint64_t VulkanEngine::getArgumentOperationDuration(const EventId id) const
{
    auto eventPointer = bufferEvents.find(id);

    if (eventPointer == bufferEvents.end())
    {
        throw std::runtime_error(std::string("Buffer event with following id does not exist or its result was already retrieved: ")
            + std::to_string(id));
    }

    if (!eventPointer->second->isValid())
    {
        bufferEvents.erase(id);
        return 0;
    }

    Logger::logDebug("Performing buffer operation synchronization for event id: " + std::to_string(id));
    eventPointer->second->wait();
    bufferEvents.erase(id);
    eventCommands.erase(id);
    stagingBuffers.erase(id);

    // todo: return correct duration
    return 0;
}

void VulkanEngine::resizeArgument(const ArgumentId id, const size_t newSize, const bool preserveData)
{
    throw std::runtime_error("Vulkan API is not yet supported");
}

void VulkanEngine::getArgumentHandle(const ArgumentId id, BufferMemory& handle)
{
    throw std::runtime_error("Vulkan API is not yet supported");
}

void VulkanEngine::addUserBuffer(UserBuffer buffer, KernelArgument& kernelArgument)
{
    throw std::runtime_error("Vulkan API is not yet supported");
}*/

ComputeActionId VulkanEngine::RunKernelAsync([[maybe_unused]] const KernelComputeData& data, [[maybe_unused]] const QueueId queueId)
{
    return 0;
}

ComputationResult VulkanEngine::WaitForComputeAction([[maybe_unused]] const ComputeActionId id)
{
    return ComputationResult();
}

void VulkanEngine::ClearData([[maybe_unused]] const KernelComputeId& id)
{

}

void VulkanEngine::ClearKernelData([[maybe_unused]] const std::string& kernelName)
{

}

ComputationResult VulkanEngine::RunKernelWithProfiling([[maybe_unused]] const KernelComputeData& data,
    [[maybe_unused]] const QueueId queueId)
{
    throw KttException("Kernel profiling is not yet supported for Vulkan backend");
}

void VulkanEngine::SetProfilingCounters([[maybe_unused]] const std::vector<std::string>& counters)
{
    throw KttException("Kernel profiling is not yet supported for Vulkan backend");
}

bool VulkanEngine::IsProfilingSessionActive([[maybe_unused]] const KernelComputeId& id)
{
    throw KttException("Kernel profiling is not yet supported for Vulkan backend");
}

uint64_t VulkanEngine::GetRemainingProfilingRuns([[maybe_unused]] const KernelComputeId& id)
{
    throw KttException("Kernel profiling is not yet supported for Vulkan backend");
}

bool VulkanEngine::HasAccurateRemainingProfilingRuns() const
{
    return false;
}

bool VulkanEngine::SupportsMultiInstanceProfiling() const
{
    return false;
}

TransferActionId VulkanEngine::UploadArgument([[maybe_unused]] KernelArgument& kernelArgument, [[maybe_unused]] const QueueId queueId)
{
    return 0;
}

TransferActionId VulkanEngine::UpdateArgument([[maybe_unused]] const ArgumentId id, [[maybe_unused]] const QueueId queueId,
    [[maybe_unused]] const void* data, [[maybe_unused]] const size_t dataSize)
{
    return 0;
}

TransferActionId VulkanEngine::DownloadArgument([[maybe_unused]] const ArgumentId id, [[maybe_unused]] const QueueId queueId,
    [[maybe_unused]] void* destination, [[maybe_unused]] const size_t dataSize)
{
    return 0;
}

TransferActionId VulkanEngine::CopyArgument([[maybe_unused]] const ArgumentId destination, [[maybe_unused]] const QueueId queueId,
    [[maybe_unused]] const ArgumentId source, [[maybe_unused]] const size_t dataSize)
{
    return 0;
}

TransferResult VulkanEngine::WaitForTransferAction([[maybe_unused]] const TransferActionId id)
{
    return TransferResult();
}

void VulkanEngine::ResizeArgument([[maybe_unused]] const ArgumentId id, [[maybe_unused]] const size_t newSize,
    [[maybe_unused]] const bool preserveData)
{

}

void VulkanEngine::GetUnifiedMemoryBufferHandle([[maybe_unused]] const ArgumentId id, [[maybe_unused]] UnifiedBufferMemory& handle)
{

}

void VulkanEngine::AddCustomBuffer([[maybe_unused]] KernelArgument& kernelArgument, [[maybe_unused]] ComputeBuffer buffer)
{

}

void VulkanEngine::ClearBuffer(const ArgumentId id)
{
    m_Buffers.erase(id);
}

void VulkanEngine::ClearBuffers()
{
    m_Buffers.clear();
}

bool VulkanEngine::HasBuffer(const ArgumentId id)
{
    return ContainsKey(m_Buffers, id);
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
}

void VulkanEngine::SynchronizeDevice()
{
    m_Device->WaitIdle();
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

void VulkanEngine::SetCompilerOptions(const std::string& options)
{
    m_CompilerOptions = options;
    ClearKernelCache();
}

void VulkanEngine::SetGlobalSizeType(const GlobalSizeType type)
{
    m_GlobalSizeType = type;
}

void VulkanEngine::SetAutomaticGlobalSizeCorrection(const bool flag)
{
    m_GlobalSizeCorrection = flag;
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

std::shared_ptr<VulkanComputePipeline> VulkanEngine::LoadPipeline([[maybe_unused]] const KernelComputeData& data)
{
    return nullptr;
}

VulkanBuffer* VulkanEngine::GetPipelineArgument([[maybe_unused]] KernelArgument& argument)
{
    return nullptr;
}

std::vector<VulkanBuffer*> VulkanEngine::GetPipelineArguments([[maybe_unused]] const std::vector<KernelArgument*>& arguments)
{
    return {};
}

std::unique_ptr<VulkanBuffer> VulkanEngine::CreateBuffer([[maybe_unused]] KernelArgument& argument)
{
    return nullptr;
}

std::unique_ptr<VulkanBuffer> VulkanEngine::CreateUserBuffer([[maybe_unused]] KernelArgument& argument,
    [[maybe_unused]] ComputeBuffer buffer)
{
    return nullptr;
}

std::vector<KernelArgument*> VulkanEngine::GetScalarArguments(const std::vector<KernelArgument*>& arguments)
{
    std::vector<KernelArgument*> result;

    for (auto* argument : arguments)
    {
        if (argument->GetMemoryType() == ArgumentMemoryType::Scalar)
        {
            result.push_back(argument);
        }
    }

    return result;
}

/*EventId VulkanEngine::enqueuePipeline(VulkanComputePipeline& pipeline, const std::vector<size_t>& globalSize, const std::vector<size_t>& localSize,
    const QueueId queue, const uint64_t kernelLaunchOverhead, const VulkanPushConstant& pushConstant)
{
    if (queue >= queues.size())
    {
        throw std::runtime_error(std::string("Invalid queue index: ") + std::to_string(queue));
    }

    std::vector<size_t> correctedGlobalSize = globalSize;
    if (globalSizeCorrection)
    {
        correctedGlobalSize = roundUpGlobalSize(correctedGlobalSize, localSize);
    }
    if (globalSizeType == GlobalSizeType::OpenCL)
    {
        correctedGlobalSize.at(0) /= localSize.at(0);
        correctedGlobalSize.at(1) /= localSize.at(1);
        correctedGlobalSize.at(2) /= localSize.at(2);
    }

    EventId eventId = nextEventId;
    auto kernelEvent = std::make_unique<VulkanEvent>(device->getDevice(), eventId, pipeline.getShaderName(), kernelLaunchOverhead);
    ++nextEventId;

    Logger::logDebug("Launching kernel " + pipeline.getShaderName() + ", event id: " + std::to_string(eventId));
    auto command = std::make_unique<VulkanCommandBufferHolder>(device->getDevice(), commandPool->getCommandPool());
    pipeline.recordDispatchShaderCommand(command->getCommandBuffer(), correctedGlobalSize, pushConstant, queryPool->getQueryPool());
    queues[queue].submitSingleCommand(command->getCommandBuffer(), kernelEvent->getFence().getFence());

    kernelEvents.insert(std::make_pair(eventId, std::move(kernelEvent)));
    eventCommands.insert(std::make_pair(eventId, std::move(command)));
    return eventId;
}

KernelResult VulkanEngine::createKernelResult(const EventId id) const
{
    auto eventPointer = kernelEvents.find(id);

    if (eventPointer == kernelEvents.end())
    {
        throw std::runtime_error(std::string("Kernel event with following id does not exist or its result was already retrieved: ")
            + std::to_string(id));
    }

    Logger::logDebug(std::string("Performing kernel synchronization for event id: ") + std::to_string(id));

    eventPointer->second->wait();
    const std::string& name = eventPointer->second->getKernelName();
    const uint64_t overhead = eventPointer->second->getOverhead();
    uint64_t duration = queryPool->getResult();

    KernelResult result(name, duration);
    result.setOverhead(overhead);

    kernelEvents.erase(id);
    eventCommands.erase(id);

    return result;
}

std::vector<VulkanBuffer*> VulkanEngine::getPipelineArguments(const std::vector<KernelArgument*>& argumentPointers)
{
    std::vector<VulkanBuffer*> result;

    for (auto* argument : argumentPointers)
    {
        if (argument->getUploadType() == ArgumentUploadType::Scalar)
        {
            continue;
        }

        if (argument->getUploadType() == ArgumentUploadType::Local)
        {
            throw std::runtime_error("Local memory arguments are currently not supported for Vulkan backend");
        }

        VulkanBuffer* existingBuffer = findBuffer(argument->getId());
        if (existingBuffer == nullptr)
        {
            uploadArgument(*argument);
            existingBuffer = findBuffer(argument->getId());
        }

        result.push_back(existingBuffer);
    }

    return result;
}*/

} // namespace ktt

#endif // KTT_API_VULKAN
