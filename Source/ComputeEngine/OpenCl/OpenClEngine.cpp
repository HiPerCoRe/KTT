#ifdef KTT_API_OPENCL

#include <ComputeEngine/OpenCl/OpenClEngine.h>
#include <ComputeEngine/OpenCl/OpenClPlatform.h>
#include <Utility/ErrorHandling/KttException.h>
#include <Utility/Logger/Logger.h>
#include <Utility/StlHelpers.h>
#include <Utility/Timer.h>

#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
#include <ComputeEngine/OpenCl/Gpa/GpaPass.h>
#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY

namespace ktt
{

OpenClEngine::OpenClEngine(const PlatformIndex platformIndex, const DeviceIndex deviceIndex, const uint32_t queueCount) :
    m_PlatformIndex(platformIndex),
    m_DeviceIndex(deviceIndex),
    m_GlobalSizeType(GlobalSizeType::OpenCL),
    m_GlobalSizeCorrection(false),
    m_KernelCache(10)
{    
    const auto platforms = OpenClPlatform::GetAllPlatforms();

    if (platformIndex >= static_cast<PlatformIndex>(platforms.size()))
    {
        throw KttException("Invalid platform index: " + std::to_string(platformIndex));
    }

    const auto& platform = platforms[static_cast<size_t>(platformIndex)];
    const auto devices = platform.GetDevices();

    if (deviceIndex >= static_cast<DeviceIndex>(devices.size()))
    {
        throw KttException("Invalid device index: " + std::to_string(deviceIndex));
    }

    const auto& device = devices[static_cast<size_t>(deviceIndex)];
    m_Context = std::make_unique<OpenClContext>(platform, device);

    for (uint32_t i = 0; i < queueCount; i++)
    {
        auto commandQueue = std::make_unique<OpenClCommandQueue>(i, *m_Context);
        m_Queues.push_back(std::move(commandQueue));
    }

    InitializeGpa();
}

OpenClEngine::OpenClEngine(const ComputeApiInitializer& initializer) :
    m_GlobalSizeType(GlobalSizeType::OpenCL),
    m_GlobalSizeCorrection(false),
    m_KernelCache(10)
{
    m_Context = std::make_unique<OpenClContext>(initializer.GetContext());

    const auto platforms = OpenClPlatform::GetAllPlatforms();

    for (size_t i = 0; i < platforms.size(); ++i)
    {
        if (m_Context->GetPlatform() == platforms[i].GetId())
        {
            m_PlatformIndex = static_cast<PlatformIndex>(i);
            break;
        }
    }

    const auto& platform = platforms[static_cast<size_t>(m_PlatformIndex)];
    const auto devices = platform.GetDevices();

    for (size_t i = 0; i < devices.size(); ++i)
    {
        if (m_Context->GetDevice() == devices[i].GetId())
        {
            m_DeviceIndex = static_cast<DeviceIndex>(i);
            break;
        }
    }

    const auto& queues = initializer.GetQueues();

    for (size_t i = 0; i < queues.size(); ++i)
    {
        auto commandQueue = std::make_unique<OpenClCommandQueue>(static_cast<QueueId>(i), *m_Context, queues[i]);
        m_Queues.push_back(std::move(commandQueue));
    }

    InitializeGpa();
}

ComputeActionId OpenClEngine::RunKernelAsync(const KernelComputeData& data, const QueueId queue)
{
    return m_Generator.GenerateComputeId();
}

KernelResult OpenClEngine::WaitForComputeAction(const ComputeActionId id) const
{
    return KernelResult();
}

KernelResult OpenClEngine::RunKernelWithProfiling(const KernelComputeData& data, const QueueId queue)
{
    return KernelResult();
}

void OpenClEngine::SetProfilingCounters(const std::vector<std::string>& counters)
{

}

bool OpenClEngine::IsProfilingSessionActive(const KernelComputeId& id)
{
    return false;
}

uint64_t OpenClEngine::GetRemainingProfilingRuns(const KernelComputeId& id)
{
#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
    if (!ContainsKey(m_GpaInstances, id))
    {
        return 0;
    }

    return static_cast<uint64_t>(m_GpaInstances[id]->GetRemainingPassCount());
#else
    throw KttException("Support for kernel profiling is not included in this version of KTT framework");
#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY
}

bool OpenClEngine::HasAccurateRemainingProfilingRuns() const
{
#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
    return true;
#else
    throw KttException("Support for kernel profiling is not included in this version of KTT framework");
#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY
}

TransferActionId OpenClEngine::UploadArgument(const KernelArgument& kernelArgument, const QueueId queue)
{
    return m_Generator.GenerateTransferId();
}

TransferActionId OpenClEngine::UpdateArgument(const ArgumentId id, const QueueId queue, const void* data,
    const size_t dataSize)
{
    return m_Generator.GenerateTransferId();
}

TransferActionId OpenClEngine::DownloadArgument(const ArgumentId id, const QueueId queue, void* destination,
    const size_t dataSize)
{
    return m_Generator.GenerateTransferId();
}

TransferActionId OpenClEngine::CopyArgument(const ArgumentId destination, const QueueId queue, const ArgumentId source,
    const size_t dataSize)
{
    return m_Generator.GenerateTransferId();
}

uint64_t OpenClEngine::WaitForTransferAction(const TransferActionId id) const
{
    return 0;
}

void OpenClEngine::ResizeArgument(const ArgumentId id, const size_t newSize, const bool preserveData)
{

}

void OpenClEngine::GetUnifiedMemoryBufferHandle(const ArgumentId id, UnifiedBufferMemory& handle)
{

}

void OpenClEngine::AddCustomBuffer(const KernelArgument& kernelArgument, ComputeBuffer buffer)
{

}

void OpenClEngine::ClearBuffer(const ArgumentId id)
{
    m_Buffers.erase(id);
}

void OpenClEngine::ClearBuffers()
{
    m_Buffers.clear();
}

QueueId OpenClEngine::GetDefaultQueue() const
{
    return static_cast<QueueId>(0);
}

std::vector<QueueId> OpenClEngine::GetAllQueues() const
{
    std::vector<QueueId> result;

    for (const auto& queue : m_Queues)
    {
        result.push_back(queue->GetId());
    }

    return result;
}

void OpenClEngine::SynchronizeQueue(const QueueId queue)
{
    if (queue >= m_Queues.size())
    {
        throw KttException("Invalid OpenCL command queue index: " + std::to_string(queue));
    }

    m_Queues[static_cast<size_t>(queue)]->Synchronize();
}

void OpenClEngine::SynchronizeDevice()
{
    for (auto& queue : m_Queues)
    {
        queue->Synchronize();
    }
}

std::vector<PlatformInfo> OpenClEngine::GetPlatformInfo() const
{
    const auto platforms = OpenClPlatform::GetAllPlatforms();
    std::vector<PlatformInfo> result;

    for (const auto& platform : platforms)
    {
        result.push_back(platform.GetInfo());
    }

    return result;
}

std::vector<DeviceInfo> OpenClEngine::GetDeviceInfo(const PlatformIndex platformIndex) const
{
    const auto platforms = OpenClPlatform::GetAllPlatforms();

    if (platformIndex >= static_cast<PlatformIndex>(platforms.size()))
    {
        throw KttException("Invalid platform index: " + std::to_string(platformIndex));
    }

    std::vector<DeviceInfo> result;
    const auto& platform = platforms[static_cast<size_t>(platformIndex)];

    for (const auto& device : platform.GetDevices())
    {
        result.push_back(device.GetInfo());
    }

    return result;
}

DeviceInfo OpenClEngine::GetCurrentDeviceInfo() const
{
    const auto& deviceInfo = GetDeviceInfo(m_PlatformIndex);
    return deviceInfo[static_cast<size_t>(m_DeviceIndex)];
}

void OpenClEngine::SetCompilerOptions(const std::string& options)
{
    OpenClProgram::SetCompilerOptions(options);
}

void OpenClEngine::SetGlobalSizeType(const GlobalSizeType type)
{
    m_GlobalSizeType = type;
}

void OpenClEngine::SetAutomaticGlobalSizeCorrection(const bool flag)
{
    m_GlobalSizeCorrection = flag;
}

void OpenClEngine::SetKernelCacheCapacity(const uint64_t capacity)
{
    m_KernelCache.SetMaxSize(static_cast<size_t>(capacity));
}

void OpenClEngine::ClearKernelCache()
{
    m_KernelCache.Clear();
}

std::shared_ptr<OpenClKernel> OpenClEngine::LoadKernel(const KernelComputeData& data)
{
    const auto id = data.GetUniqueIdentifier();

    if (m_KernelCache.GetMaxSize() > 0 && m_KernelCache.Exists(id))
    {
        return m_KernelCache.Get(id)->second;
    }

    auto program = std::make_unique<OpenClProgram>(*m_Context, data.GetSource());
    program->Build();
    auto kernel = std::make_shared<OpenClKernel>(std::move(program), data.GetName());

    if (m_KernelCache.GetMaxSize() > 0)
    {
        m_KernelCache.Put(id, kernel);
    }

    return kernel;
}

void OpenClEngine::SetKernelArguments(OpenClKernel& kernel, const KernelComputeData& data)
{
    const auto& arguments = data.GetArguments();
    kernel.ResetArguments();

    for (const auto* argument : arguments)
    {
        SetKernelArgument(kernel, *argument);
    }
}

void OpenClEngine::SetKernelArgument(OpenClKernel& kernel, const KernelArgument& argument)
{
    if (argument.GetMemoryLocation() == ArgumentMemoryLocation::Undefined)
    {
        kernel.SetArgument(argument);
        return;
    }

    const auto id = argument.GetId();

    if (!ContainsKey(m_Buffers, id))
    {
        throw KttException("Buffer corresponding to kernel argument with id " + std::to_string(id) + " was not found");
    }

    kernel.SetArgument(*m_Buffers[id]);
}

//EventId OpenCLEngine::runKernelAsync(const KernelRuntimeData& kernelData, const std::vector<KernelArgument*>& argumentPointers, const QueueId queue)
//{
//    Timer overheadTimer;
//    overheadTimer.start();
//
//    OpenCLKernel* kernel;
//    std::unique_ptr<OpenCLKernel> kernelUnique;
//    std::unique_ptr<OpenCLProgram> program;
//
//    if (kernelCacheFlag)
//    {
//        if (kernelCache.find(std::make_pair(kernelData.getName(), kernelData.getSource())) == kernelCache.end())
//        {
//            if (kernelCache.size() >= kernelCacheCapacity)
//            {
//                clearKernelCache();
//            }
//            std::unique_ptr<OpenCLProgram> cacheProgram = createAndBuildProgram(kernelData.getSource());
//            auto cacheKernel = std::make_unique<OpenCLKernel>(context->getDevice(), cacheProgram->getProgram(), kernelData.getName());
//            kernelCache.insert(std::make_pair(std::make_pair(kernelData.getName(), kernelData.getSource()), std::make_pair(std::move(cacheKernel),
//                std::move(cacheProgram))));
//        }
//        auto cachePointer = kernelCache.find(std::make_pair(kernelData.getName(), kernelData.getSource()));
//        kernel = cachePointer->second.first.get();
//    }
//    else
//    {
//        program = createAndBuildProgram(kernelData.getSource());
//        kernelUnique = std::make_unique<OpenCLKernel>(context->getDevice(), program->getProgram(), kernelData.getName());
//        kernel = kernelUnique.get();
//    }
//
//    checkLocalMemoryModifiers(argumentPointers, kernelData.getLocalMemoryModifiers());
//    kernel->resetKernelArguments();
//
//    for (const auto argument : argumentPointers)
//    {
//        if (argument->getUploadType() == ArgumentUploadType::Local)
//        {
//            setKernelArgument(*kernel, *argument, kernelData.getLocalMemoryModifiers());
//        }
//        else
//        {
//            setKernelArgument(*kernel, *argument);
//        }
//    }
//
//    overheadTimer.stop();
//
//    return enqueueKernel(*kernel, kernelData.getGlobalSize(), kernelData.getLocalSize(), queue, overheadTimer.getElapsedTime());
//}
//
//KernelResult OpenCLEngine::getKernelResult(const EventId id, const std::vector<OutputDescriptor>& outputDescriptors) const
//{
//    KernelResult result = createKernelResult(id);
//
//    for (const auto& descriptor : outputDescriptors)
//    {
//        downloadArgument(descriptor.getArgumentId(), descriptor.getOutputDestination(), descriptor.getOutputSizeInBytes());
//    }
//
//    return result;
//}
//
//uint64_t OpenCLEngine::getKernelOverhead(const EventId id) const
//{
//    auto eventPointer = kernelEvents.find(id);
//
//    if (eventPointer == kernelEvents.end())
//    {
//        throw std::runtime_error(std::string("Kernel event with following id does not exist or its result was already retrieved: ")
//            + std::to_string(id));
//    }
//
//    return eventPointer->second->getOverhead();
//}
//
//void OpenCLEngine::clearEvents()
//{
//    kernelEvents.clear();
//    bufferEvents.clear();
//
//#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
//    kernelToEventMap.clear();
//
//    for (const auto& profilingInstance : kernelProfilingInstances)
//    {
//        // There is currently no way to abort active profiling session without performing all passes, so dummy passes are launched
//        while (profilingInstance.second->getRemainingPassCount() > 0)
//        {
//            launchDummyPass(profilingInstance.first.first, profilingInstance.first.second);
//        }
//
//        profilingInstance.second->generateProfilingData();
//    }
//
//    kernelProfilingInstances.clear();
//#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY
//}
//
//uint64_t OpenCLEngine::uploadArgument(KernelArgument& kernelArgument)
//{
//    if (kernelArgument.getUploadType() != ArgumentUploadType::Vector)
//    {
//        return 0;
//    }
//
//    EventId eventId = uploadArgumentAsync(kernelArgument, getDefaultQueue());
//    return getArgumentOperationDuration(eventId);
//}
//
//EventId OpenCLEngine::uploadArgumentAsync(KernelArgument& kernelArgument, const QueueId queue)
//{
//    if (queue >= commandQueues.size())
//    {
//        throw std::runtime_error(std::string("Invalid queue index: ") + std::to_string(queue));
//    }
//
//    if (findBuffer(kernelArgument.getId()) != nullptr)
//    {
//        throw std::runtime_error(std::string("Buffer with following id already exists: ") + std::to_string(kernelArgument.getId()));
//    }
//
//    if (kernelArgument.getUploadType() != ArgumentUploadType::Vector)
//    {
//        return UINT64_MAX;
//    }
//
//    Logger::logDebug("Uploading buffer for argument " + std::to_string(kernelArgument.getId()) + ", event id: " + std::to_string(nextEventId));
//
//    const EventId eventId = nextEventId;
//    const ArgumentMemoryLocation location = kernelArgument.getMemoryLocation();
//    auto buffer = std::make_unique<OpenCLBuffer>(context->getContext(), kernelArgument);
//
//    if (location == ArgumentMemoryLocation::Unified || location == ArgumentMemoryLocation::HostZeroCopy)
//    {
//        bufferEvents.insert(std::make_pair(eventId, std::make_unique<OpenCLEvent>(eventId, false)));
//    }
//    else
//    {
//        auto profilingEvent = std::make_unique<OpenCLEvent>(eventId, true);
//        buffer->uploadData(commandQueues.at(queue)->getQueue(), kernelArgument.getData(), kernelArgument.getDataSizeInBytes(),
//            profilingEvent->getEvent());
//
//        profilingEvent->setReleaseFlag();
//        bufferEvents.insert(std::make_pair(eventId, std::move(profilingEvent)));
//    }
//
//    buffers.insert(std::move(buffer)); // buffer data will be stolen
//    nextEventId++;
//    return eventId;
//}
//
//uint64_t OpenCLEngine::updateArgument(const ArgumentId id, const void* data, const size_t dataSizeInBytes)
//{
//    EventId eventId = updateArgumentAsync(id, data, dataSizeInBytes, getDefaultQueue());
//    return getArgumentOperationDuration(eventId);
//}
//
//EventId OpenCLEngine::updateArgumentAsync(const ArgumentId id, const void* data, const size_t dataSizeInBytes, const QueueId queue)
//{
//    if (queue >= commandQueues.size())
//    {
//        throw std::runtime_error(std::string("Invalid queue index: ") + std::to_string(queue));
//    }
//
//    OpenCLBuffer* buffer = findBuffer(id);
//
//    if (buffer == nullptr)
//    {
//        throw std::runtime_error(std::string("Buffer with following id was not found: ") + std::to_string(id));
//    }
//
//    EventId eventId = nextEventId;
//    auto profilingEvent = std::make_unique<OpenCLEvent>(eventId, true);
//    
//    Logger::getLogger().log(LoggingLevel::Debug, "Updating buffer for argument " + std::to_string(id) + ", event id: " + std::to_string(eventId));
//
//    if (dataSizeInBytes == 0)
//    {
//        buffer->uploadData(commandQueues.at(queue)->getQueue(), data, buffer->getBufferSize(), profilingEvent->getEvent());
//    }
//    else
//    {
//        buffer->uploadData(commandQueues.at(queue)->getQueue(), data, dataSizeInBytes, profilingEvent->getEvent());
//    }
//
//    profilingEvent->setReleaseFlag();
//    bufferEvents.insert(std::make_pair(eventId, std::move(profilingEvent)));
//    nextEventId++;
//    return eventId;
//}
//
//uint64_t OpenCLEngine::downloadArgument(const ArgumentId id, void* destination, const size_t dataSizeInBytes) const
//{
//    EventId eventId = downloadArgumentAsync(id, destination, dataSizeInBytes, getDefaultQueue());
//    return getArgumentOperationDuration(eventId);
//}
//
//EventId OpenCLEngine::downloadArgumentAsync(const ArgumentId id, void* destination, const size_t dataSizeInBytes, const QueueId queue) const
//{
//    if (queue >= commandQueues.size())
//    {
//        throw std::runtime_error(std::string("Invalid queue index: ") + std::to_string(queue));
//    }
//
//    OpenCLBuffer* buffer = findBuffer(id);
//
//    if (buffer == nullptr)
//    {
//        throw std::runtime_error(std::string("Buffer with following id was not found: ") + std::to_string(id));
//    }
//
//    EventId eventId = nextEventId;
//    auto profilingEvent = std::make_unique<OpenCLEvent>(eventId, true);
//
//    Logger::getLogger().log(LoggingLevel::Debug, "Downloading buffer for argument " + std::to_string(id) + ", event id: " + std::to_string(eventId));
//
//    if (dataSizeInBytes == 0)
//    {
//        buffer->downloadData(commandQueues.at(queue)->getQueue(), destination, buffer->getBufferSize(), profilingEvent->getEvent());
//    }
//    else
//    {
//        buffer->downloadData(commandQueues.at(queue)->getQueue(), destination, dataSizeInBytes, profilingEvent->getEvent());
//    }
//
//    profilingEvent->setReleaseFlag();
//    bufferEvents.insert(std::make_pair(eventId, std::move(profilingEvent)));
//    nextEventId++;
//    return eventId;
//}
//
//KernelArgument OpenCLEngine::downloadArgumentObject(const ArgumentId id, uint64_t* downloadDuration) const
//{
//    OpenCLBuffer* buffer = findBuffer(id);
//
//    if (buffer == nullptr)
//    {
//        throw std::runtime_error(std::string("Buffer with following id was not found: ") + std::to_string(id));
//    }
//
//    KernelArgument argument(buffer->getKernelArgumentId(), buffer->getBufferSize() / buffer->getElementSize(), buffer->getElementSize(),
//        buffer->getDataType(), buffer->getMemoryLocation(), buffer->getAccessType(), ArgumentUploadType::Vector);
//
//    bool validEvent = true;
//
//    if (buffer->getMemoryLocation() == ArgumentMemoryLocation::Unified)
//    {
//        validEvent = false;
//    }
//
//    EventId eventId = nextEventId;
//    auto profilingEvent = std::make_unique<OpenCLEvent>(eventId, validEvent);
//
//    Logger::getLogger().log(LoggingLevel::Debug, "Downloading buffer for argument " + std::to_string(id) + ", event id: " + std::to_string(eventId));
//    buffer->downloadData(commandQueues.at(getDefaultQueue())->getQueue(), argument.getData(), argument.getDataSizeInBytes(),
//        profilingEvent->getEvent());
//
//    if (validEvent)
//    {
//        profilingEvent->setReleaseFlag();
//    }
//    
//    bufferEvents.insert(std::make_pair(eventId, std::move(profilingEvent)));
//    nextEventId++;
//
//    uint64_t duration = getArgumentOperationDuration(eventId);
//    if (downloadDuration != nullptr)
//    {
//        *downloadDuration = duration;
//    }
//    
//    return argument;
//}
//
//uint64_t OpenCLEngine::copyArgument(const ArgumentId destination, const ArgumentId source, const size_t dataSizeInBytes)
//{
//    EventId eventId = copyArgumentAsync(destination, source, dataSizeInBytes, getDefaultQueue());
//    return getArgumentOperationDuration(eventId);
//}
//
//EventId OpenCLEngine::copyArgumentAsync(const ArgumentId destination, const ArgumentId source, const size_t dataSizeInBytes, const QueueId queue)
//{
//    if (queue >= commandQueues.size())
//    {
//        throw std::runtime_error(std::string("Invalid queue index: ") + std::to_string(queue));
//    }
//
//    OpenCLBuffer* destinationBuffer = findBuffer(destination);
//    OpenCLBuffer* sourceBuffer = findBuffer(source);
//
//    if (destinationBuffer == nullptr || sourceBuffer == nullptr)
//    {
//        throw std::runtime_error(std::string("One of the buffers with following ids does not exist: ") + std::to_string(destination) + ", "
//            + std::to_string(source));
//    }
//
//    if (sourceBuffer->getDataType() != destinationBuffer->getDataType())
//    {
//        throw std::runtime_error("Data type for buffers during copying operation must match");
//    }
//
//    EventId eventId = nextEventId;
//    auto profilingEvent = std::make_unique<OpenCLEvent>(eventId, true);
//
//    Logger::getLogger().log(LoggingLevel::Debug, "Copying buffer for argument " + std::to_string(source) + " into buffer for argument "
//        + std::to_string(destination) + ", event id: " + std::to_string(eventId));
//
//    if (dataSizeInBytes == 0)
//    {
//        destinationBuffer->uploadData(commandQueues.at(queue)->getQueue(), sourceBuffer->getBuffer(), sourceBuffer->getBufferSize(),
//            profilingEvent->getEvent());
//    }
//    else
//    {
//        destinationBuffer->uploadData(commandQueues.at(queue)->getQueue(), sourceBuffer->getBuffer(), dataSizeInBytes, profilingEvent->getEvent());
//    }
//
//    profilingEvent->setReleaseFlag();
//    bufferEvents.insert(std::make_pair(eventId, std::move(profilingEvent)));
//    nextEventId++;
//    return eventId;
//}
//
//uint64_t OpenCLEngine::persistArgument(KernelArgument& kernelArgument, const bool flag)
//{
//    bool bufferFound = false;
//    auto iterator = persistentBuffers.cbegin();
//
//    while (iterator != persistentBuffers.cend())
//    {
//        if (iterator->get()->getKernelArgumentId() == kernelArgument.getId())
//        {
//            bufferFound = true;
//            if (!flag)
//            {
//                persistentBuffers.erase(iterator);
//            }
//            break;
//        }
//        else
//        {
//            ++iterator;
//        }
//    }
//    
//    if (flag && !bufferFound)
//    {
//        Logger::logDebug("Uploading persistent buffer for argument " + std::to_string(kernelArgument.getId()) + ", event id: "
//            + std::to_string(nextEventId));
//        
//        const EventId eventId = nextEventId;
//        const ArgumentMemoryLocation location = kernelArgument.getMemoryLocation();
//        auto buffer = std::make_unique<OpenCLBuffer>(context->getContext(), kernelArgument);
//
//        if (location == ArgumentMemoryLocation::Unified || location == ArgumentMemoryLocation::HostZeroCopy)
//        {
//            bufferEvents.insert(std::make_pair(eventId, std::make_unique<OpenCLEvent>(eventId, false)));
//        }
//        else
//        {
//            auto profilingEvent = std::make_unique<OpenCLEvent>(eventId, true);
//            buffer->uploadData(commandQueues.at(getDefaultQueue())->getQueue(), kernelArgument.getData(), kernelArgument.getDataSizeInBytes(),
//                profilingEvent->getEvent());
//
//            profilingEvent->setReleaseFlag();
//            bufferEvents.insert(std::make_pair(eventId, std::move(profilingEvent)));
//        }
//
//        persistentBuffers.insert(std::move(buffer)); // buffer data will be stolen
//        nextEventId++;
//
//        return getArgumentOperationDuration(eventId);
//    }
//
//    return 0;
//}
//
//uint64_t OpenCLEngine::getArgumentOperationDuration(const EventId id) const
//{
//    auto eventPointer = bufferEvents.find(id);
//
//    if (eventPointer == bufferEvents.end())
//    {
//        throw std::runtime_error(std::string("Buffer event with following id does not exist or its result was already retrieved: ")
//            + std::to_string(id));
//    }
//
//    if (!eventPointer->second->isValid())
//    {
//        bufferEvents.erase(id);
//        return 0;
//    }
//
//    Logger::getLogger().log(LoggingLevel::Debug, "Performing buffer operation synchronization for event id: " + std::to_string(id));
//
//    checkOpenCLError(clWaitForEvents(1, eventPointer->second->getEvent()), "clWaitForEvents");
//    cl_ulong duration = eventPointer->second->getEventCommandDuration();
//    bufferEvents.erase(id);
//
//    return static_cast<uint64_t>(duration);
//}
//
//void OpenCLEngine::resizeArgument(const ArgumentId id, const size_t newSize, const bool preserveData)
//{
//    OpenCLBuffer* buffer = findBuffer(id);
//
//    if (buffer == nullptr)
//    {
//        throw std::runtime_error(std::string("Buffer with following id was not found: ") + std::to_string(id));
//    }
//
//    Logger::getLogger().log(LoggingLevel::Debug, "Resizing buffer for argument " + std::to_string(id));
//    buffer->resize(commandQueues.at(getDefaultQueue())->getQueue(), newSize, preserveData);
//}
//
//void OpenCLEngine::getArgumentHandle(const ArgumentId id, BufferMemory& handle)
//{
//    OpenCLBuffer* buffer = findBuffer(id);
//
//    if (buffer == nullptr)
//    {
//        throw std::runtime_error(std::string("Buffer with following id was not found: ") + std::to_string(id));
//    }
//
//    handle = buffer->getRawBuffer();
//}
//
//void OpenCLEngine::addUserBuffer(UserBuffer buffer, KernelArgument& kernelArgument)
//{
//    if (findBuffer(kernelArgument.getId()) != nullptr)
//    {
//        throw std::runtime_error(std::string("User buffer with the following id already exists: ") + std::to_string(kernelArgument.getId()));
//    }
//
//    auto openclBuffer = std::make_unique<OpenCLBuffer>(buffer, kernelArgument);
//    userBuffers.insert(std::move(openclBuffer));
//}
//
//void OpenCLEngine::initializeKernelProfiling(const KernelRuntimeData& kernelData)
//{
//    #if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
//    initializeKernelProfiling(kernelData.getName(), kernelData.getSource());
//    #else
//    throw std::runtime_error("Support for kernel profiling is not included in this version of KTT framework");
//    #endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY
//}
//
//EventId OpenCLEngine::runKernelWithProfiling(const KernelRuntimeData& kernelData, const std::vector<KernelArgument*>& argumentPointers,
//    const QueueId queue)
//{
//    #if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
//    Timer overheadTimer;
//    overheadTimer.start();
//
//    OpenCLKernel* kernel;
//    std::unique_ptr<OpenCLKernel> kernelUnique;
//    std::unique_ptr<OpenCLProgram> program;
//
//    if (kernelCacheFlag)
//    {
//        if (kernelCache.find(std::make_pair(kernelData.getName(), kernelData.getSource())) == kernelCache.end())
//        {
//            if (kernelCache.size() >= kernelCacheCapacity)
//            {
//                clearKernelCache();
//            }
//            std::unique_ptr<OpenCLProgram> cacheProgram = createAndBuildProgram(kernelData.getSource());
//            auto cacheKernel = std::make_unique<OpenCLKernel>(context->getDevice(), cacheProgram->getProgram(), kernelData.getName());
//            kernelCache.insert(std::make_pair(std::make_pair(kernelData.getName(), kernelData.getSource()), std::make_pair(std::move(cacheKernel),
//                std::move(cacheProgram))));
//        }
//        auto cachePointer = kernelCache.find(std::make_pair(kernelData.getName(), kernelData.getSource()));
//        kernel = cachePointer->second.first.get();
//    }
//    else
//    {
//        program = createAndBuildProgram(kernelData.getSource());
//        kernelUnique = std::make_unique<OpenCLKernel>(context->getDevice(), program->getProgram(), kernelData.getName());
//        kernel = kernelUnique.get();
//    }
//
//    checkLocalMemoryModifiers(argumentPointers, kernelData.getLocalMemoryModifiers());
//    kernel->resetKernelArguments();
//
//    for (const auto argument : argumentPointers)
//    {
//        if (argument->getUploadType() == ArgumentUploadType::Local)
//        {
//            setKernelArgument(*kernel, *argument, kernelData.getLocalMemoryModifiers());
//        }
//        else
//        {
//            setKernelArgument(*kernel, *argument);
//        }
//    }
//
//    overheadTimer.stop();
//
//    if (kernelProfilingInstances.find({kernelData.getName(), kernelData.getSource()}) == kernelProfilingInstances.end())
//    {
//        initializeKernelProfiling(kernelData.getName(), kernelData.getSource());
//    }
//
//    auto profilingInstance = kernelProfilingInstances.find({kernelData.getName(), kernelData.getSource()});
//    auto profilingPass = std::make_unique<GPAProfilingPass>(gpaInterface->getFunctionTable(), *profilingInstance->second.get());
//    EventId id = enqueueKernel(*kernel, kernelData.getGlobalSize(), kernelData.getLocalSize(), queue, overheadTimer.getElapsedTime());
//    kernelToEventMap[std::make_pair(kernelData.getName(), kernelData.getSource())].push_back(id);
//    
//    auto eventPointer = kernelEvents.find(id);
//    Logger::logDebug(std::string("Performing kernel synchronization for event id: ") + std::to_string(id));
//    checkOpenCLError(clWaitForEvents(1, eventPointer->second->getEvent()), "clWaitForEvents");
//
//    return id;
//    #else
//    throw std::runtime_error("Support for kernel profiling is not included in this version of KTT framework");
//    #endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY
//}
//
//KernelResult OpenCLEngine::getKernelResultWithProfiling(const EventId id, const std::vector<OutputDescriptor>& outputDescriptors)
//{
//    #if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
//    KernelResult result = createKernelResult(id);
//
//    for (const auto& descriptor : outputDescriptors)
//    {
//        downloadArgument(descriptor.getArgumentId(), descriptor.getOutputDestination(), descriptor.getOutputSizeInBytes());
//    }
//
//    const std::pair<std::string, std::string>& kernelKey = getKernelFromEvent(id);
//    auto profilingInstance = kernelProfilingInstances.find(kernelKey);
//    if (profilingInstance == kernelProfilingInstances.end())
//    {
//        throw std::runtime_error(std::string("No profiling data exists for the following kernel in current configuration: " + kernelKey.first));
//    }
//
//    KernelProfilingData profilingData = profilingInstance->second->generateProfilingData();
//    result.setProfilingData(profilingData);
//
//    kernelProfilingInstances.erase(kernelKey);
//    const std::vector<EventId>& eventIds = kernelToEventMap.find(kernelKey)->second;
//    for (const auto eventId : eventIds)
//    {
//        kernelEvents.erase(eventId);
//    }
//    kernelToEventMap.erase(kernelKey);
//
//    return result;
//    #else
//    throw std::runtime_error("Support for kernel profiling is not included in this version of KTT framework");
//    #endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY
//}
//
//void OpenCLEngine::setKernelProfilingCounters(const std::vector<std::string>& counterNames)
//{
//    #if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
//    gpaProfilingContext->setCounters(counterNames);
//    #else
//    throw std::runtime_error("Support for kernel profiling is not included in this version of KTT framework");
//    #endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY
//}

void OpenClEngine::InitializeGpa()
{
#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
    m_GpaInterface = std::make_unique<GpaInterface>();
    m_GpaContext = std::make_unique<GpaContext>(m_GpaInterface->GetFunctions(), *m_Queues[GetDefaultQueue()]);
#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY
}

//EventId OpenCLEngine::enqueueKernel(OpenCLKernel& kernel, const std::vector<size_t>& globalSize, const std::vector<size_t>& localSize,
//    const QueueId queue, const uint64_t kernelLaunchOverhead) const
//{
//    if (queue >= commandQueues.size())
//    {
//        throw std::runtime_error(std::string("Invalid queue index: ") + std::to_string(queue));
//    }
//
//    std::vector<size_t> correctedGlobalSize = globalSize;
//    if (globalSizeType != GlobalSizeType::OpenCL)
//    {
//        correctedGlobalSize.at(0) *= localSize.at(0);
//        correctedGlobalSize.at(1) *= localSize.at(1);
//        correctedGlobalSize.at(2) *= localSize.at(2);
//    }
//    if (globalSizeCorrection)
//    {
//        correctedGlobalSize = roundUpGlobalSize(correctedGlobalSize, localSize);
//    }
//
//    EventId eventId = nextEventId;
//    auto profilingEvent = std::make_unique<OpenCLEvent>(eventId, kernel.getKernelName(), kernelLaunchOverhead, kernel.getCompilationData());
//    nextEventId++;
//
//    Logger::getLogger().log(LoggingLevel::Debug, "Launching kernel " + kernel.getKernelName() + ", event id: " + std::to_string(eventId));
//    cl_int result = clEnqueueNDRangeKernel(commandQueues.at(queue)->getQueue(), kernel.getKernel(),
//        static_cast<cl_uint>(correctedGlobalSize.size()), nullptr, correctedGlobalSize.data(), localSize.data(), 0, nullptr, profilingEvent->getEvent());
//    checkOpenCLError(result, "clEnqueueNDRangeKernel");
//
//    profilingEvent->setReleaseFlag();
//    kernelEvents.insert(std::make_pair(eventId, std::move(profilingEvent)));
//    return eventId;
//}
//
//KernelResult OpenCLEngine::createKernelResult(const EventId id) const
//{
//    auto eventPointer = kernelEvents.find(id);
//
//    if (eventPointer == kernelEvents.end())
//    {
//        throw std::runtime_error(std::string("Kernel event with following id does not exist or its result was already retrieved: ")
//            + std::to_string(id));
//    }
//
//    Logger::getLogger().log(LoggingLevel::Debug, std::string("Performing kernel synchronization for event id: ") + std::to_string(id));
//
//    checkOpenCLError(clWaitForEvents(1, eventPointer->second->getEvent()), "clWaitForEvents");
//    std::string name = eventPointer->second->getKernelName();
//    cl_ulong duration = eventPointer->second->getEventCommandDuration();
//    uint64_t overhead = eventPointer->second->getOverhead();
//    KernelCompilationData compilationData = eventPointer->second->getCompilationData();
//    kernelEvents.erase(id);
//
//    KernelResult result(name, static_cast<uint64_t>(duration));
//    result.setOverhead(overhead);
//    result.setCompilationData(compilationData);
//
//    return result;
//}
//
//#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
//
//void OpenCLEngine::initializeKernelProfiling(const std::string& kernelName, const std::string& kernelSource)
//{
//    auto profilingInstance = kernelProfilingInstances.find(std::make_pair(kernelName, kernelSource));
//    if (profilingInstance == kernelProfilingInstances.end())
//    {
//        kernelProfilingInstances.insert({std::make_pair(kernelName, kernelSource),
//            std::make_unique<GPAProfilingInstance>(gpaInterface->getFunctionTable(), *gpaProfilingContext.get())});
//        kernelToEventMap.insert({std::make_pair(kernelName, kernelSource), std::vector<EventId>{}});
//    }
//}
//
//const std::pair<std::string, std::string>& OpenCLEngine::getKernelFromEvent(const EventId id) const
//{
//    for (const auto& entry : kernelToEventMap)
//    {
//        if (containsElement(entry.second, id))
//        {
//            return entry.first;
//        }
//    }
//
//    throw std::runtime_error(std::string("Corresponding kernel was not found for event with id: ") + std::to_string(id));
//}
//
//void OpenCLEngine::launchDummyPass(const std::string& kernelName, const std::string& kernelSource)
//{
//    auto profilingInstance = kernelProfilingInstances.find({kernelName, kernelSource});
//    auto profilingPass = std::make_unique<GPAProfilingPass>(gpaInterface->getFunctionTable(), *profilingInstance->second);
//}
//
//#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY

} // namespace ktt

#endif // KTT_PLATFORM_OPENCL
