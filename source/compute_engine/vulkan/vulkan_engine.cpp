#ifdef KTT_PLATFORM_VULKAN

#include <limits>
#include <compute_engine/vulkan/vulkan_engine.h>
#include <utility/ktt_utility.h>
#include <utility/logger.h>
#include <utility/timer.h>

namespace ktt
{

VulkanEngine::VulkanEngine(const DeviceIndex deviceIndex, const uint32_t queueCount) :
    deviceIndex(deviceIndex),
    queueCount(queueCount),
    compilerOptions(std::string("")),
    globalSizeType(GlobalSizeType::Vulkan),
    globalSizeCorrection(false),
    kernelCacheFlag(true),
    kernelCacheCapacity(10),
    persistentBufferFlag(true),
    nextEventId(0)
{
    std::vector<const char*> instanceExtensions;
    std::vector<const char*> validationLayers;

    #ifdef KTT_CONFIGURATION_DEBUG
    instanceExtensions.emplace_back("VK_EXT_debug_report");
    validationLayers.emplace_back("VK_LAYER_LUNARG_standard_validation");
    #endif

    Logger::logDebug("Initializing Vulkan instance");
    instance = std::make_unique<VulkanInstance>("KTT Compute Engine", instanceExtensions, validationLayers);

    std::vector<VulkanPhysicalDevice> devices = instance->getPhysicalDevices();
    if (deviceIndex >= devices.size())
    {
        throw std::runtime_error(std::string("Invalid device index: ") + std::to_string(deviceIndex));
    }

    Logger::logDebug("Initializing Vulkan device and queues");
    device = std::make_unique<VulkanDevice>(devices.at(deviceIndex), queueCount, VK_QUEUE_COMPUTE_BIT, std::vector<const char*>{}, validationLayers);
    queues = device->getQueues();

    Logger::logDebug("Initializing Vulkan command pool");
    commandPool = std::make_unique<VulkanCommandPool>(device->getDevice(), device->getQueueFamilyIndex());

    Logger::logDebug("Initializing Vulkan query pool");
    queryPool = std::make_unique<VulkanQueryPool>(device->getDevice(), devices.at(deviceIndex).getProperties().limits.timestampPeriod);
}

KernelResult VulkanEngine::runKernel(const KernelRuntimeData& kernelData, const std::vector<KernelArgument*>& argumentPointers,
    const std::vector<OutputDescriptor>& outputDescriptors)
{
    EventId eventId = runKernelAsync(kernelData, argumentPointers, getDefaultQueue());
    KernelResult result = getKernelResult(eventId, outputDescriptors);
    return result;
}

EventId VulkanEngine::runKernelAsync(const KernelRuntimeData& kernelData, const std::vector<KernelArgument*>& argumentPointers, const QueueId queue)
{
    Timer overheadTimer;
    overheadTimer.start();

    VulkanComputePipeline* pipeline;
    std::unique_ptr<VulkanComputePipeline> pipelineUnique;
    std::unique_ptr<VulkanDescriptorSetLayout> layout;
    std::unique_ptr<VulkanShaderModule> shader;
    const uint32_t bindingCount = static_cast<uint32_t>(kernelData.getArgumentIds().size());

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
                cacheShader->getShaderModule(), kernelData.getName());
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
            kernelData.getName());
        pipeline = pipelineUnique.get();
    }

    std::vector<VulkanBuffer*> pipelineArguments = getPipelineArguments(argumentPointers);
    pipeline->bindArguments(pipelineArguments);
    overheadTimer.stop();

    return enqueuePipeline(*pipeline, kernelData.getGlobalSize(), kernelData.getLocalSize(), queue, overheadTimer.getElapsedTime());
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

void VulkanEngine::setCompilerOptions(const std::string& options)
{
    compilerOptions = options;
}

void VulkanEngine::setGlobalSizeType(const GlobalSizeType type)
{
    globalSizeType = type;
}

void VulkanEngine::setAutomaticGlobalSizeCorrection(const bool flag)
{
    globalSizeCorrection = flag;
}

void VulkanEngine::setKernelCacheUsage(const bool flag)
{
    if (!flag)
    {
        clearKernelCache();
    }
    kernelCacheFlag = flag;
}

void VulkanEngine::setKernelCacheCapacity(const size_t capacity)
{
    kernelCacheCapacity = capacity;
}

void VulkanEngine::clearKernelCache()
{
    pipelineCache.clear();
}

QueueId VulkanEngine::getDefaultQueue() const
{
    return 0;
}

std::vector<QueueId> VulkanEngine::getAllQueues() const
{
    std::vector<QueueId> result;

    for (size_t i = 0; i < queues.size(); ++i)
    {
        result.push_back(static_cast<QueueId>(i));
    }

    return result;
}

void VulkanEngine::synchronizeQueue(const QueueId queue)
{
    if (queue >= queues.size())
    {
        throw std::runtime_error(std::string("Invalid queue index: ") + std::to_string(queue));
    }

    queues[queue].waitIdle();
}

void VulkanEngine::synchronizeDevice()
{
    device->waitIdle();
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

uint64_t VulkanEngine::updateArgument(const ArgumentId id, const void* data, const size_t dataSizeInBytes)
{
    EventId eventId = updateArgumentAsync(id, data, dataSizeInBytes, getDefaultQueue());
    return getArgumentOperationDuration(eventId);
}

EventId VulkanEngine::updateArgumentAsync(const ArgumentId id, const void* data, const size_t dataSizeInBytes, const QueueId queue)
{
    throw std::runtime_error("Vulkan API is not yet supported");
}

uint64_t VulkanEngine::downloadArgument(const ArgumentId id, void* destination, const size_t dataSizeInBytes) const
{
    EventId eventId = downloadArgumentAsync(id, destination, dataSizeInBytes, getDefaultQueue());
    return getArgumentOperationDuration(eventId);
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

void VulkanEngine::setPersistentBufferUsage(const bool flag)
{
    persistentBufferFlag = flag;
}

void VulkanEngine::clearBuffer(const ArgumentId id)
{
    auto iterator = buffers.cbegin();

    while (iterator != buffers.cend())
    {
        if (iterator->get()->getKernelArgumentId() == id)
        {
            buffers.erase(iterator);
            return;
        }
        else
        {
            ++iterator;
        }
    }
}

void VulkanEngine::clearBuffers()
{
    buffers.clear();
}

void VulkanEngine::clearBuffers(const ArgumentAccessType accessType)
{
    auto iterator = buffers.cbegin();

    while (iterator != buffers.cend())
    {
        if (iterator->get()->getAccessType() == accessType)
        {
            iterator = buffers.erase(iterator);
        }
        else
        {
            ++iterator;
        }
    }
}

void VulkanEngine::printComputeAPIInfo(std::ostream& outputTarget) const
{
    outputTarget << "Platform 0: " << "Vulkan" << std::endl;
    std::vector<VulkanPhysicalDevice> devices = instance->getPhysicalDevices();

    for (size_t i = 0; i < devices.size(); i++)
    {
        outputTarget << "Device " << i << ": " << devices.at(i).getName() << std::endl;
    }
    outputTarget << std::endl;
}

std::vector<PlatformInfo> VulkanEngine::getPlatformInfo() const
{
    PlatformInfo info(0, "Vulkan");
    info.setVendor("N/A");
    info.setVersion(instance->getAPIVersion());

    std::vector<std::string> extensions = instance->getExtensions();
    std::string mergedExtensions("");
    for (size_t i = 0; i < extensions.size(); ++i)
    {
        mergedExtensions += extensions[i];
        if (i != extensions.size() - 1)
        {
            mergedExtensions += ", ";
        }
    }
    info.setExtensions(mergedExtensions);
    return std::vector<PlatformInfo>{info};
}

std::vector<DeviceInfo> VulkanEngine::getDeviceInfo(const PlatformIndex) const
{
    std::vector<DeviceInfo> result;
    std::vector<VulkanPhysicalDevice> devices = instance->getPhysicalDevices();

    for (const auto& device : devices)
    {
        result.push_back(device.getDeviceInfo());
    }

    return result;
}

DeviceInfo VulkanEngine::getCurrentDeviceInfo() const
{
    return getDeviceInfo(0).at(deviceIndex);
}

void VulkanEngine::initializeKernelProfiling(const KernelRuntimeData&)
{
    throw std::runtime_error("Kernel profiling is not supported for Vulkan backend");
}

EventId VulkanEngine::runKernelWithProfiling(const KernelRuntimeData&, const std::vector<KernelArgument*>&, const QueueId)
{
    throw std::runtime_error("Kernel profiling is not supported for Vulkan backend");
}

uint64_t VulkanEngine::getRemainingKernelProfilingRuns(const std::string&, const std::string&)
{
    throw std::runtime_error("Kernel profiling is not supported for Vulkan backend");
}

KernelResult VulkanEngine::getKernelResultWithProfiling(const EventId, const std::vector<OutputDescriptor>&)
{
    throw std::runtime_error("Kernel profiling is not supported for Vulkan backend");
}

void VulkanEngine::setKernelProfilingCounters(const std::vector<std::string>&)
{
    throw std::runtime_error("Kernel profiling is not supported for Vulkan backend");
}

EventId VulkanEngine::enqueuePipeline(VulkanComputePipeline& pipeline, const std::vector<size_t>& globalSize, const std::vector<size_t>& localSize,
    const QueueId queue, const uint64_t kernelLaunchOverhead)
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
    pipeline.recordDispatchShaderCommand(command->getCommandBuffer(), correctedGlobalSize, queryPool->getQueryPool());
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
        if (argument->getUploadType() == ArgumentUploadType::Local || argument->getUploadType() == ArgumentUploadType::Scalar)
        {
            throw std::runtime_error("Only vector arguments are currently supported for Vulkan backend");
        }
        else if (argument->getUploadType() == ArgumentUploadType::Vector)
        {
            VulkanBuffer* existingBuffer = findBuffer(argument->getId());
            if (existingBuffer == nullptr)
            {
                uploadArgument(*argument);
                existingBuffer = findBuffer(argument->getId());
            }

            result.push_back(existingBuffer);
        }
    }

    return result;
}

VulkanBuffer* VulkanEngine::findBuffer(const ArgumentId id) const
{
    if (persistentBufferFlag)
    {
        for (const auto& buffer : persistentBuffers)
        {
            if (buffer->getKernelArgumentId() == id)
            {
                return buffer.get();
            }
        }
    }

    for (const auto& buffer : buffers)
    {
        if (buffer->getKernelArgumentId() == id)
        {
            return buffer.get();
        }
    }

    return nullptr;
}

} // namespace ktt

#endif // KTT_PLATFORM_VULKAN
