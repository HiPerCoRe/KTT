#ifdef KTT_PLATFORM_CUDA

#include <stdexcept>
#include <compute_engine/cuda/cuda_engine.h>
#include <utility/ktt_utility.h>
#include <utility/logger.h>
#include <utility/timer.h>

namespace ktt
{

CUDAEngine::CUDAEngine(const DeviceIndex deviceIndex, const uint32_t queueCount) :
    deviceIndex(deviceIndex),
    queueCount(queueCount),
    compilerOptions(std::string("--gpu-architecture=compute_30")),
    globalSizeType(GlobalSizeType::CUDA),
    globalSizeCorrection(false),
    kernelCacheFlag(true),
    kernelCacheCapacity(10),
    persistentBufferFlag(true),
    nextEventId(0)
{
    Logger::logDebug("Initializing CUDA runtime");
    checkCUDAError(cuInit(0), "cuInit");

    auto devices = getCUDADevices();
    if (deviceIndex >= devices.size())
    {
        throw std::runtime_error(std::string("Invalid device index: ") + std::to_string(deviceIndex));
    }

    Logger::logDebug("Initializing CUDA context");
    context = std::make_unique<CUDAContext>(devices.at(deviceIndex).getDevice());

    Logger::logDebug("Initializing CUDA streams");
    for (uint32_t i = 0; i < queueCount; i++)
    {
        auto stream = std::make_unique<CUDAStream>(i, context->getContext(), devices.at(deviceIndex).getDevice());
        streams.push_back(std::move(stream));
    }

#ifdef KTT_PROFILING
    Logger::logDebug("Initializing CUPTI profiling metric IDs");
    const std::vector<std::string>& metricNames = getDefaultProfilingMetricNames();
    profilingMetrics = getProfilingMetricsForCurrentDevice(metricNames);
#endif // KTT_PROFILING
}

KernelResult CUDAEngine::runKernel(const KernelRuntimeData& kernelData, const std::vector<KernelArgument*>& argumentPointers,
    const std::vector<OutputDescriptor>& outputDescriptors)
{
    EventId eventId = runKernelAsync(kernelData, argumentPointers, getDefaultQueue());
    KernelResult result = getKernelResult(eventId, outputDescriptors);
    return result;
}

EventId CUDAEngine::runKernelAsync(const KernelRuntimeData& kernelData, const std::vector<KernelArgument*>& argumentPointers, const QueueId queue)
{
    Timer overheadTimer;
    overheadTimer.start();

    CUDAKernel* kernel;
    std::unique_ptr<CUDAKernel> kernelUnique;

    if (kernelCacheFlag)
    {
        if (kernelCache.find(std::make_pair(kernelData.getName(), kernelData.getSource())) == kernelCache.end())
        {
            if (kernelCache.size() >= kernelCacheCapacity)
            {
                clearKernelCache();
            }
            std::unique_ptr<CUDAProgram> program = createAndBuildProgram(kernelData.getSource());
            auto kernel = std::make_unique<CUDAKernel>(program->getPtxSource(), kernelData.getName());
            kernelCache.insert(std::make_pair(std::make_pair(kernelData.getName(), kernelData.getSource()), std::move(kernel)));
        }
        auto cachePointer = kernelCache.find(std::make_pair(kernelData.getName(), kernelData.getSource()));
        kernel = cachePointer->second.get();
    }
    else
    {
        std::unique_ptr<CUDAProgram> program = createAndBuildProgram(kernelData.getSource());
        kernelUnique = std::make_unique<CUDAKernel>(program->getPtxSource(), kernelData.getName());
        kernel = kernelUnique.get();
    }

    std::vector<CUdeviceptr*> kernelArguments = getKernelArguments(argumentPointers);

    overheadTimer.stop();

    return enqueueKernel(*kernel, kernelData.getGlobalSize(), kernelData.getLocalSize(), kernelArguments,
        getSharedMemorySizeInBytes(argumentPointers, kernelData.getLocalMemoryModifiers()), queue, overheadTimer.getElapsedTime());
}

KernelResult CUDAEngine::getKernelResult(const EventId id, const std::vector<OutputDescriptor>& outputDescriptors) const
{
    KernelResult result = createKernelResult(id);

    for (const auto& descriptor : outputDescriptors)
    {
        downloadArgument(descriptor.getArgumentId(), descriptor.getOutputDestination(), descriptor.getOutputSizeInBytes());
    }

    return result;
}

uint64_t CUDAEngine::getKernelOverhead(const EventId id) const
{
    auto eventPointer = kernelEvents.find(id);

    if (eventPointer == kernelEvents.end())
    {
        throw std::runtime_error(std::string("Kernel event with following id does not exist or its result was already retrieved: ")
            + std::to_string(id));
    }

    return eventPointer->second.first->getOverhead();
}

void CUDAEngine::setCompilerOptions(const std::string& options)
{
    compilerOptions = options;
}

void CUDAEngine::setGlobalSizeType(const GlobalSizeType type)
{
    globalSizeType = type;
}

void CUDAEngine::setAutomaticGlobalSizeCorrection(const bool flag)
{
    globalSizeCorrection = flag;
}

void CUDAEngine::setKernelCacheUsage(const bool flag)
{
    if (!flag)
    {
        clearKernelCache();
    }
    kernelCacheFlag = flag;
}

void CUDAEngine::setKernelCacheCapacity(const size_t capacity)
{
    kernelCacheCapacity = capacity;
}

void CUDAEngine::clearKernelCache()
{
    kernelCache.clear();
}

QueueId CUDAEngine::getDefaultQueue() const
{
    return 0;
}

std::vector<QueueId> CUDAEngine::getAllQueues() const
{
    std::vector<QueueId> result;

    for (size_t i = 0; i < streams.size(); i++)
    {
        result.push_back(static_cast<QueueId>(i));
    }

    return result;
}

void CUDAEngine::synchronizeQueue(const QueueId queue)
{
    if (queue >= streams.size())
    {
        throw std::runtime_error(std::string("Invalid stream index: ") + std::to_string(queue));
    }

    checkCUDAError(cuStreamSynchronize(streams.at(queue)->getStream()), "cuStreamSynchronize");
}

void CUDAEngine::synchronizeDevice()
{
    for (auto& stream : streams)
    {
        checkCUDAError(cuStreamSynchronize(stream->getStream()), "cuStreamSynchronize");
    }
}

void CUDAEngine::clearEvents()
{
    kernelEvents.clear();
    bufferEvents.clear();
}

uint64_t CUDAEngine::uploadArgument(KernelArgument& kernelArgument)
{
    if (kernelArgument.getUploadType() != ArgumentUploadType::Vector)
    {
        return 0;
    }

    EventId eventId = uploadArgumentAsync(kernelArgument, getDefaultQueue());
    return getArgumentOperationDuration(eventId);
}

EventId CUDAEngine::uploadArgumentAsync(KernelArgument& kernelArgument, const QueueId queue)
{
    if (queue >= streams.size())
    {
        throw std::runtime_error(std::string("Invalid stream index: ") + std::to_string(queue));
    }

    if (findBuffer(kernelArgument.getId()) != nullptr)
    {
        throw std::runtime_error(std::string("Buffer with following id already exists: ") + std::to_string(kernelArgument.getId()));
    }

    if (kernelArgument.getUploadType() != ArgumentUploadType::Vector)
    {
        return UINT64_MAX;
    }
    
    std::unique_ptr<CUDABuffer> buffer = nullptr;
    EventId eventId = nextEventId;

    Logger::getLogger().log(LoggingLevel::Debug, "Uploading buffer for argument " + std::to_string(kernelArgument.getId()) + ", event id: "
        + std::to_string(eventId));

    if (kernelArgument.getMemoryLocation() == ArgumentMemoryLocation::HostZeroCopy)
    {
        buffer = std::make_unique<CUDABuffer>(kernelArgument, true);
        bufferEvents.insert(std::make_pair(eventId, std::make_pair(std::make_unique<CUDAEvent>(eventId, false),
            std::make_unique<CUDAEvent>(eventId, false))));
    }
    else
    {
        buffer = std::make_unique<CUDABuffer>(kernelArgument, false);
        auto startEvent = std::make_unique<CUDAEvent>(eventId, true);
        auto endEvent = std::make_unique<CUDAEvent>(eventId, true);
        buffer->uploadData(streams.at(queue)->getStream(), kernelArgument.getData(), kernelArgument.getDataSizeInBytes(), startEvent->getEvent(),
            endEvent->getEvent());
        bufferEvents.insert(std::make_pair(eventId, std::make_pair(std::move(startEvent), std::move(endEvent))));
    }

    buffers.insert(std::move(buffer)); // buffer data will be stolen
    nextEventId++;
    return eventId;
}

uint64_t CUDAEngine::updateArgument(const ArgumentId id, const void* data, const size_t dataSizeInBytes)
{
    EventId eventId = updateArgumentAsync(id, data, dataSizeInBytes, getDefaultQueue());
    return getArgumentOperationDuration(eventId);
}

EventId CUDAEngine::updateArgumentAsync(const ArgumentId id, const void* data, const size_t dataSizeInBytes, const QueueId queue)
{
    if (queue >= streams.size())
    {
        throw std::runtime_error(std::string("Invalid stream index: ") + std::to_string(queue));
    }

    CUDABuffer* buffer = findBuffer(id);

    if (buffer == nullptr)
    {
        throw std::runtime_error(std::string("Buffer with following id was not found: ") + std::to_string(id));
    }

    EventId eventId = nextEventId;
    auto startEvent = std::make_unique<CUDAEvent>(eventId, true);
    auto endEvent = std::make_unique<CUDAEvent>(eventId, true);

    Logger::getLogger().log(LoggingLevel::Debug, "Updating buffer for argument " + std::to_string(id) + ", event id: " + std::to_string(eventId));

    if (dataSizeInBytes == 0)
    {
        buffer->uploadData(streams.at(queue)->getStream(), data, buffer->getBufferSize(), startEvent->getEvent(), endEvent->getEvent());
    }
    else
    {
        buffer->uploadData(streams.at(queue)->getStream(), data, dataSizeInBytes, startEvent->getEvent(), endEvent->getEvent());
    }

    bufferEvents.insert(std::make_pair(eventId, std::make_pair(std::move(startEvent), std::move(endEvent))));
    nextEventId++;
    return eventId;
}

uint64_t CUDAEngine::downloadArgument(const ArgumentId id, void* destination, const size_t dataSizeInBytes) const
{
    EventId eventId = downloadArgumentAsync(id, destination, dataSizeInBytes, getDefaultQueue());
    return getArgumentOperationDuration(eventId);
}

EventId CUDAEngine::downloadArgumentAsync(const ArgumentId id, void* destination, const size_t dataSizeInBytes, const QueueId queue) const
{
    if (queue >= streams.size())
    {
        throw std::runtime_error(std::string("Invalid stream index: ") + std::to_string(queue));
    }

    CUDABuffer* buffer = findBuffer(id);

    if (buffer == nullptr)
    {
        throw std::runtime_error(std::string("Buffer with following id was not found: ") + std::to_string(id));
    }

    EventId eventId = nextEventId;
    auto startEvent = std::make_unique<CUDAEvent>(eventId, true);
    auto endEvent = std::make_unique<CUDAEvent>(eventId, true);

    Logger::getLogger().log(LoggingLevel::Debug, "Downloading buffer for argument " + std::to_string(id) + ", event id: " + std::to_string(eventId));

    if (dataSizeInBytes == 0)
    {
        buffer->downloadData(streams.at(queue)->getStream(), destination, buffer->getBufferSize(), startEvent->getEvent(), endEvent->getEvent());
    }
    else
    {
        buffer->downloadData(streams.at(queue)->getStream(), destination, dataSizeInBytes, startEvent->getEvent(), endEvent->getEvent());
    }

    bufferEvents.insert(std::make_pair(eventId, std::make_pair(std::move(startEvent), std::move(endEvent))));
    nextEventId++;
    return eventId;
}

KernelArgument CUDAEngine::downloadArgumentObject(const ArgumentId id, uint64_t* downloadDuration) const
{
    CUDABuffer* buffer = findBuffer(id);

    if (buffer == nullptr)
    {
        throw std::runtime_error(std::string("Buffer with following id was not found: ") + std::to_string(id));
    }

    KernelArgument argument(buffer->getKernelArgumentId(), buffer->getBufferSize() / buffer->getElementSize(), buffer->getElementSize(),
        buffer->getDataType(), buffer->getMemoryLocation(), buffer->getAccessType(), ArgumentUploadType::Vector);
    
    EventId eventId = nextEventId;
    auto startEvent = std::make_unique<CUDAEvent>(eventId, true);
    auto endEvent = std::make_unique<CUDAEvent>(eventId, true);

    Logger::getLogger().log(LoggingLevel::Debug, "Downloading buffer for argument " + std::to_string(id) + ", event id: " + std::to_string(eventId));
    buffer->downloadData(streams.at(getDefaultQueue())->getStream(), argument.getData(), argument.getDataSizeInBytes(), startEvent->getEvent(),
        endEvent->getEvent());

    bufferEvents.insert(std::make_pair(eventId, std::make_pair(std::move(startEvent), std::move(endEvent))));
    nextEventId++;

    uint64_t duration = getArgumentOperationDuration(eventId);
    if (downloadDuration != nullptr)
    {
        *downloadDuration = duration;
    }

    return argument;
}

uint64_t CUDAEngine::copyArgument(const ArgumentId destination, const ArgumentId source, const size_t dataSizeInBytes)
{
    EventId eventId = copyArgumentAsync(destination, source, dataSizeInBytes, getDefaultQueue());
    return getArgumentOperationDuration(eventId);
}

EventId CUDAEngine::copyArgumentAsync(const ArgumentId destination, const ArgumentId source, const size_t dataSizeInBytes, const QueueId queue)
{
    if (queue >= streams.size())
    {
        throw std::runtime_error(std::string("Invalid stream index: ") + std::to_string(queue));
    }

    CUDABuffer* destinationBuffer = findBuffer(destination);
    CUDABuffer* sourceBuffer = findBuffer(source);

    if (destinationBuffer == nullptr || sourceBuffer == nullptr)
    {
        throw std::runtime_error(std::string("One of the buffers with following ids does not exist: ") + std::to_string(destination) + ", "
            + std::to_string(source));
    }

    if (sourceBuffer->getDataType() != destinationBuffer->getDataType())
    {
        throw std::runtime_error("Data type for buffers during copying operation must match");
    }

    EventId eventId = nextEventId;
    auto startEvent = std::make_unique<CUDAEvent>(eventId, true);
    auto endEvent = std::make_unique<CUDAEvent>(eventId, true);

    Logger::getLogger().log(LoggingLevel::Debug, "Copying buffer for argument " + std::to_string(source) + " into buffer for argument "
        + std::to_string(destination) + ", event id: " + std::to_string(eventId));

    if (dataSizeInBytes == 0)
    {
        destinationBuffer->uploadData(streams.at(queue)->getStream(), sourceBuffer, sourceBuffer->getBufferSize(), startEvent->getEvent(),
            endEvent->getEvent());
    }
    else
    {
        destinationBuffer->uploadData(streams.at(queue)->getStream(), sourceBuffer, dataSizeInBytes, startEvent->getEvent(), endEvent->getEvent());
    }

    bufferEvents.insert(std::make_pair(eventId, std::make_pair(std::move(startEvent), std::move(endEvent))));
    nextEventId++;
    return eventId;
}

uint64_t CUDAEngine::persistArgument(KernelArgument& kernelArgument, const bool flag)
{
    bool bufferFound = false;
    auto iterator = persistentBuffers.cbegin();

    while (iterator != persistentBuffers.cend())
    {
        if (iterator->get()->getKernelArgumentId() == kernelArgument.getId())
        {
            bufferFound = true;
            if (!flag)
            {
                persistentBuffers.erase(iterator);
            }
            break;
        }
        else
        {
            ++iterator;
        }
    }
    
    if (flag && !bufferFound)
    {
        std::unique_ptr<CUDABuffer> buffer = nullptr;
        EventId eventId = nextEventId;

        Logger::getLogger().log(LoggingLevel::Debug, "Uploading persistent buffer for argument " + std::to_string(kernelArgument.getId())
            + ", event id: " + std::to_string(eventId));

        if (kernelArgument.getMemoryLocation() == ArgumentMemoryLocation::HostZeroCopy)
        {
            buffer = std::make_unique<CUDABuffer>(kernelArgument, true);
            bufferEvents.insert(std::make_pair(eventId, std::make_pair(std::make_unique<CUDAEvent>(eventId, false),
                std::make_unique<CUDAEvent>(eventId, false))));
        }
        else
        {
            buffer = std::make_unique<CUDABuffer>(kernelArgument, false);
            auto startEvent = std::make_unique<CUDAEvent>(eventId, true);
            auto endEvent = std::make_unique<CUDAEvent>(eventId, true);
            buffer->uploadData(streams.at(getDefaultQueue())->getStream(), kernelArgument.getData(), kernelArgument.getDataSizeInBytes(),
                startEvent->getEvent(), endEvent->getEvent());
            bufferEvents.insert(std::make_pair(eventId, std::make_pair(std::move(startEvent), std::move(endEvent))));
        }

        persistentBuffers.insert(std::move(buffer)); // buffer data will be stolen
        nextEventId++;

        return getArgumentOperationDuration(eventId);
    }

    return 0;
}

uint64_t CUDAEngine::getArgumentOperationDuration(const EventId id) const
{
    auto eventPointer = bufferEvents.find(id);

    if (eventPointer == bufferEvents.end())
    {
        throw std::runtime_error(std::string("Buffer event with following id does not exist or its result was already retrieved: ")
            + std::to_string(id));
    }

    if (!eventPointer->second.first->isValid())
    {
        bufferEvents.erase(id);
        return 0;
    }

    Logger::getLogger().log(LoggingLevel::Debug, "Performing buffer operation synchronization for event id: " + std::to_string(id));

    // Wait until the second event in pair (the end event) finishes
    checkCUDAError(cuEventSynchronize(eventPointer->second.second->getEvent()), "cuEventSynchronize");
    float duration = getEventCommandDuration(eventPointer->second.first->getEvent(), eventPointer->second.second->getEvent());
    bufferEvents.erase(id);

    return static_cast<uint64_t>(duration);
}

void CUDAEngine::resizeArgument(const ArgumentId id, const size_t newSize, const bool preserveData)
{
    CUDABuffer* buffer = findBuffer(id);

    if (buffer == nullptr)
    {
        throw std::runtime_error(std::string("Buffer with following id was not found: ") + std::to_string(id));
    }

    Logger::getLogger().log(LoggingLevel::Debug, "Resizing buffer for argument " + std::to_string(id));
    buffer->resize(newSize, preserveData);
}

void CUDAEngine::clearBuffer(const ArgumentId id)
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

void CUDAEngine::setPersistentBufferUsage(const bool flag)
{
    persistentBufferFlag = flag;
}

void CUDAEngine::clearBuffers()
{
    buffers.clear();
}

void CUDAEngine::clearBuffers(const ArgumentAccessType accessType)
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

void CUDAEngine::printComputeAPIInfo(std::ostream& outputTarget) const
{
    outputTarget << "Platform 0: " << "NVIDIA CUDA" << std::endl;
    auto devices = getCUDADevices();

    for (size_t i = 0; i < devices.size(); i++)
    {
        outputTarget << "Device " << i << ": " << devices.at(i).getName() << std::endl;
    }
    outputTarget << std::endl;
}

std::vector<PlatformInfo> CUDAEngine::getPlatformInfo() const
{
    int driverVersion;
    checkCUDAError(cuDriverGetVersion(&driverVersion), "cuDriverGetVersion");

    PlatformInfo cuda(0, "NVIDIA CUDA");
    cuda.setVendor("NVIDIA Corporation");
    cuda.setVersion(std::to_string(driverVersion));
    cuda.setExtensions("N/A");
    return std::vector<PlatformInfo>{cuda};
}

std::vector<DeviceInfo> CUDAEngine::getDeviceInfo(const PlatformIndex) const
{
    std::vector<DeviceInfo> result;
    auto devices = getCUDADevices();

    for (size_t i = 0; i < devices.size(); i++)
    {
        result.push_back(getCUDADeviceInfo(static_cast<DeviceIndex>(i)));
    }

    return result;
}

DeviceInfo CUDAEngine::getCurrentDeviceInfo() const
{
    return getCUDADeviceInfo(deviceIndex);
}

void CUDAEngine::initializeKernelProfiling(const KernelRuntimeData& kernelData)
{
    #ifdef KTT_PROFILING
    initializeKernelProfiling(kernelData.getName(), kernelData.getSource());
    #else
    throw std::runtime_error("Support for kernel profiling is not included in this version of KTT framework");
    #endif // KTT_PROFILING
}

EventId CUDAEngine::runKernelWithProfiling(const KernelRuntimeData& kernelData, const std::vector<KernelArgument*>& argumentPointers,
    const QueueId queue)
{
    #ifdef KTT_PROFILING
    Timer overheadTimer;
    overheadTimer.start();

    CUDAKernel* kernel;
    std::unique_ptr<CUDAKernel> kernelUnique;

    if (kernelCacheFlag)
    {
        if (kernelCache.find(std::make_pair(kernelData.getName(), kernelData.getSource())) == kernelCache.end())
        {
            if (kernelCache.size() >= kernelCacheCapacity)
            {
                clearKernelCache();
            }
            std::unique_ptr<CUDAProgram> program = createAndBuildProgram(kernelData.getSource());
            auto kernel = std::make_unique<CUDAKernel>(program->getPtxSource(), kernelData.getName());
            kernelCache.insert(std::make_pair(std::make_pair(kernelData.getName(), kernelData.getSource()), std::move(kernel)));
        }
        auto cachePointer = kernelCache.find(std::make_pair(kernelData.getName(), kernelData.getSource()));
        kernel = cachePointer->second.get();
    }
    else
    {
        std::unique_ptr<CUDAProgram> program = createAndBuildProgram(kernelData.getSource());
        kernelUnique = std::make_unique<CUDAKernel>(program->getPtxSource(), kernelData.getName());
        kernel = kernelUnique.get();
    }

    std::vector<CUdeviceptr*> kernelArguments = getKernelArguments(argumentPointers);

    overheadTimer.stop();

    if (kernelProfilingStates.find(std::make_pair(kernelData.getName(), kernelData.getSource())) == kernelProfilingStates.end())
    {
        initializeKernelProfiling(kernelData.getName(), kernelData.getSource());
    }

    auto profilingState = kernelProfilingStates.find(std::make_pair(kernelData.getName(), kernelData.getSource()));
    CUpti_SubscriberHandle subscriber;
    bool firstRun = false;
    if (!profilingState->second.hasValidKernelDuration())
    {
        firstRun = true;
    }
    else
    {
        std::vector<CUDAProfilingMetric>* metricData = profilingState->second.getProfilingMetrics();
        checkCUDAError(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)getMetricValueCallback, metricData), "cuptiSubscribe");
        checkCUDAError(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunch), "cuptiEnableCallback");
        checkCUDAError(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel), "cuptiEnableCallback");
    }

    EventId id = enqueueKernel(*kernel, kernelData.getGlobalSize(), kernelData.getLocalSize(), kernelArguments,
        getSharedMemorySizeInBytes(argumentPointers, kernelData.getLocalMemoryModifiers()), queue, overheadTimer.getElapsedTime());
    kernelToEventMap[std::make_pair(kernelData.getName(), kernelData.getSource())].push_back(id);

    if (firstRun)
    {
        auto eventPointer = kernelEvents.find(id);
        Logger::logDebug("Performing kernel synchronization for event id: " + std::to_string(id));
        checkCUDAError(cuEventSynchronize(eventPointer->second.second->getEvent()), "cuEventSynchronize");
        float duration = getEventCommandDuration(eventPointer->second.first->getEvent(), eventPointer->second.second->getEvent());
        profilingState->second.updateState(static_cast<uint64_t>(duration));
    }
    else
    {
        checkCUDAError(cuptiUnsubscribe(subscriber), "cuptiUnsubscribe");
        profilingState->second.updateState();
    }

    return id;
    #else
    throw std::runtime_error("Support for kernel profiling is not included in this version of KTT framework");
    #endif // KTT_PROFILING
}

uint64_t CUDAEngine::getRemainingKernelProfilingRuns(const std::string& kernelName, const std::string& kernelSource)
{
    #ifdef KTT_PROFILING
    if (kernelProfilingStates.find(std::make_pair(kernelName, kernelSource)) == kernelProfilingStates.end())
    {
        return 0;
    }

    return kernelProfilingStates.find(std::make_pair(kernelName, kernelSource))->second.getRemainingKernelRuns();
    #else
    throw std::runtime_error("Support for kernel profiling is not included in this version of KTT framework");
    #endif // KTT_PROFILING
}

KernelResult CUDAEngine::getKernelResultWithProfiling(const EventId id, const std::vector<OutputDescriptor>& outputDescriptors)
{
    #ifdef KTT_PROFILING
    KernelResult result = createKernelResult(id);

    for (const auto& descriptor : outputDescriptors)
    {
        downloadArgument(descriptor.getArgumentId(), descriptor.getOutputDestination(), descriptor.getOutputSizeInBytes());
    }

    const std::pair<std::string, std::string>& kernelKey = getKernelFromEvent(id);
    auto profilingState = kernelProfilingStates.find(kernelKey);
    if (profilingState == kernelProfilingStates.end())
    {
        throw std::runtime_error(std::string("No profiling data exists for the following kernel in current configuration: " + kernelKey.first));
    }

    KernelProfilingData profilingData = profilingState->second.generateProfilingData();
    result.setProfilingData(profilingData);

    kernelProfilingStates.erase(kernelKey);
    const std::vector<EventId>& eventIds = kernelToEventMap.find(kernelKey)->second;
    for (const auto eventId : eventIds)
    {
        kernelEvents.erase(eventId);
    }
    kernelToEventMap.erase(kernelKey);

    return result;
    #else
    throw std::runtime_error("Support for kernel profiling is not included in this version of KTT framework");
    #endif // KTT_PROFILING
}

void CUDAEngine::setKernelProfilingCounters(const std::vector<std::string>& counterNames)
{
    #ifdef KTT_PROFILING
    profilingMetrics.clear();
    profilingMetrics = getProfilingMetricsForCurrentDevice(counterNames);
    #else
    throw std::runtime_error("Support for kernel profiling is not included in this version of KTT framework");
    #endif // KTT_PROFILING
}

std::unique_ptr<CUDAProgram> CUDAEngine::createAndBuildProgram(const std::string& source) const
{
    auto program = std::make_unique<CUDAProgram>(source);
    program->build(compilerOptions);
    return program;
}

EventId CUDAEngine::enqueueKernel(CUDAKernel& kernel, const std::vector<size_t>& globalSize, const std::vector<size_t>& localSize,
    const std::vector<CUdeviceptr*>& kernelArguments, const size_t localMemorySize, const QueueId queue, const uint64_t kernelLaunchOverhead)
{
    if (queue >= streams.size())
    {
        throw std::runtime_error(std::string("Invalid stream index: ") + std::to_string(queue));
    }

    std::vector<void*> kernelArgumentsVoid;
    for (size_t i = 0; i < kernelArguments.size(); i++)
    {
        kernelArgumentsVoid.push_back((void*)kernelArguments.at(i));
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
    auto startEvent = std::make_unique<CUDAEvent>(eventId, kernel.getKernelName(), kernelLaunchOverhead);
    auto endEvent = std::make_unique<CUDAEvent>(eventId, kernel.getKernelName(), kernelLaunchOverhead);
    nextEventId++;

    Logger::getLogger().log(LoggingLevel::Debug, "Launching kernel " + kernel.getKernelName() + ", event id: " + std::to_string(eventId));
    checkCUDAError(cuEventRecord(startEvent->getEvent(), streams.at(queue)->getStream()), "cuEventRecord");
    checkCUDAError(cuLaunchKernel(kernel.getKernel(), static_cast<unsigned int>(correctedGlobalSize.at(0)),
        static_cast<unsigned int>(correctedGlobalSize.at(1)), static_cast<unsigned int>(correctedGlobalSize.at(2)),
        static_cast<unsigned int>(localSize.at(0)), static_cast<unsigned int>(localSize.at(1)), static_cast<unsigned int>(localSize.at(2)),
        static_cast<unsigned int>(localMemorySize), streams.at(queue)->getStream(), kernelArgumentsVoid.data(), nullptr),
        "cuLaunchKernel");
    checkCUDAError(cuEventRecord(endEvent->getEvent(), streams.at(queue)->getStream()), "cuEventRecord");

    kernelEvents.insert(std::make_pair(eventId, std::make_pair(std::move(startEvent), std::move(endEvent))));
    return eventId;
}

KernelResult CUDAEngine::createKernelResult(const EventId id) const
{
    auto eventPointer = kernelEvents.find(id);

    if (eventPointer == kernelEvents.end())
    {
        throw std::runtime_error(std::string("Kernel event with following id does not exist or its result was already retrieved: ")
            + std::to_string(id));
    }

    Logger::getLogger().log(LoggingLevel::Debug, std::string("Performing kernel synchronization for event id: ") + std::to_string(id));

    // Wait until the second event in pair (the end event) finishes
    checkCUDAError(cuEventSynchronize(eventPointer->second.second->getEvent()), "cuEventSynchronize");
    std::string name = eventPointer->second.first->getKernelName();
    float duration = getEventCommandDuration(eventPointer->second.first->getEvent(), eventPointer->second.second->getEvent());
    uint64_t overhead = eventPointer->second.first->getOverhead();
    kernelEvents.erase(id);

    KernelResult result(name, static_cast<uint64_t>(duration));
    result.setOverhead(overhead);

    return result;
}

DeviceInfo CUDAEngine::getCUDADeviceInfo(const DeviceIndex deviceIndex) const
{
    auto devices = getCUDADevices();
    DeviceInfo result(deviceIndex, devices.at(deviceIndex).getName());

    CUdevice id = devices.at(deviceIndex).getDevice();
    result.setExtensions("N/A");
    result.setVendor("NVIDIA Corporation");
    
    size_t globalMemory;
    checkCUDAError(cuDeviceTotalMem(&globalMemory, id), "cuDeviceTotalMem");
    result.setGlobalMemorySize(globalMemory);

    int localMemory;
    checkCUDAError(cuDeviceGetAttribute(&localMemory, CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK, id), "cuDeviceGetAttribute");
    result.setLocalMemorySize(localMemory);

    int constantMemory;
    checkCUDAError(cuDeviceGetAttribute(&constantMemory, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, id), "cuDeviceGetAttribute");
    result.setMaxConstantBufferSize(constantMemory);

    int computeUnits;
    checkCUDAError(cuDeviceGetAttribute(&computeUnits, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, id), "cuDeviceGetAttribute");
    result.setMaxComputeUnits(computeUnits);

    int workGroupSize;
    checkCUDAError(cuDeviceGetAttribute(&workGroupSize, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, id), "cuDeviceGetAttribute");
    result.setMaxWorkGroupSize(workGroupSize);
    result.setDeviceType(DeviceType::GPU);

    return result;
}

std::vector<CUDADevice> CUDAEngine::getCUDADevices() const
{
    int deviceCount;
    checkCUDAError(cuDeviceGetCount(&deviceCount), "cuDeviceGetCount");

    std::vector<CUdevice> deviceIds(deviceCount);
    for (int i = 0; i < deviceCount; i++)
    {
        checkCUDAError(cuDeviceGet(&deviceIds.at(i), i), "cuDeviceGet");
    }

    std::vector<CUDADevice> devices;
    for (const auto deviceId : deviceIds)
    {
        char name[40];
        checkCUDAError(cuDeviceGetName(name, 40, deviceId), "cuDeviceGetName");
        devices.push_back(CUDADevice(deviceId, std::string(name)));
    }

    return devices;
}

std::vector<CUdeviceptr*> CUDAEngine::getKernelArguments(const std::vector<KernelArgument*>& argumentPointers)
{
    std::vector<CUdeviceptr*> result;

    for (const auto argument : argumentPointers)
    {
        if (argument->getUploadType() == ArgumentUploadType::Local)
        {
            continue;
        }
        else if (argument->getUploadType() == ArgumentUploadType::Vector)
        {
            CUdeviceptr* cachedBuffer = loadBufferFromCache(argument->getId());
            if (cachedBuffer == nullptr)
            {
                uploadArgument(*argument);
                cachedBuffer = loadBufferFromCache(argument->getId());
            }

            result.push_back(cachedBuffer);
        }
        else if (argument->getUploadType() == ArgumentUploadType::Scalar)
        {
            result.push_back((CUdeviceptr*)argument->getData());
        }
    }

    return result;
}

size_t CUDAEngine::getSharedMemorySizeInBytes(const std::vector<KernelArgument*>& argumentPointers,
    const std::vector<LocalMemoryModifier>& modifiers) const
{
    for (const auto& modifier : modifiers)
    {
        bool modifierArgumentFound = false;

        for (const auto argument : argumentPointers)
        {
            if (modifier.getArgument() == argument->getId() && argument->getUploadType() == ArgumentUploadType::Local)
            {
                modifierArgumentFound = true;
            }
        }

        if (!modifierArgumentFound)
        {
            throw std::runtime_error(std::string("No matching local memory argument found for modifier, argument id in modifier: ")
                + std::to_string(modifier.getArgument()));
        }
    }

    size_t result = 0;

    for (const auto argument : argumentPointers)
    {
        if (argument->getUploadType() != ArgumentUploadType::Local)
        {
            continue;
        }

        size_t numberOfElements = argument->getNumberOfElements();

        for (const auto& modifier : modifiers)
        {
            if (modifier.getArgument() != argument->getId())
            {
                continue;
            }

            numberOfElements = modifier.getModifiedSize(numberOfElements);
        }

        size_t argumentSize = argument->getElementSizeInBytes() * numberOfElements;
        result += argumentSize;
    }

    return result;
}

CUDABuffer* CUDAEngine::findBuffer(const ArgumentId id) const
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

CUdeviceptr* CUDAEngine::loadBufferFromCache(const ArgumentId id) const
{
    CUDABuffer* buffer = findBuffer(id);

    if (buffer != nullptr)
    {
        return buffer->getBuffer();
    }

    return nullptr;
}

#ifdef KTT_PROFILING

void CUDAEngine::initializeKernelProfiling(const std::string& kernelName, const std::string& kernelSource)
{
    auto profilingState = kernelProfilingStates.find(std::make_pair(kernelName, kernelSource));
    if (profilingState == kernelProfilingStates.end())
    {
        kernelProfilingStates.insert(std::make_pair(std::make_pair(kernelName, kernelSource), CUDAProfilingState(context->getContext(),
            context->getDevice(), profilingMetrics)));
        kernelToEventMap.insert(std::make_pair(std::make_pair(kernelName, kernelSource), std::vector<EventId>{}));
    }
}

const std::pair<std::string, std::string>& CUDAEngine::getKernelFromEvent(const EventId id) const
{
    for (const auto& entry : kernelToEventMap)
    {
        if (elementExists(id, entry.second))
        {
            return entry.first;
        }
    }

    throw std::runtime_error(std::string("Corresponding kernel was not found for event with id: ") + std::to_string(id));
}

CUpti_MetricID CUDAEngine::getMetricIdFromName(const std::string& metricName)
{
    CUpti_MetricID metricId;
    const CUptiResult result = cuptiMetricGetIdFromName(context->getDevice(), metricName.c_str(), &metricId);

    switch (result)
    {
    case CUPTI_SUCCESS:
        return metricId;
    case CUPTI_ERROR_INVALID_METRIC_NAME:
        return std::numeric_limits<CUpti_MetricID>::max();
    default:
        checkCUDAError(result, "cuptiMetricGetIdFromName");
    }

    return 0;
}

std::vector<std::pair<std::string, CUpti_MetricID>> CUDAEngine::getProfilingMetricsForCurrentDevice(const std::vector<std::string>& metricNames)
{
    std::vector<std::pair<std::string, CUpti_MetricID>> collectedMetrics;

    for (const auto& metricName : metricNames)
    {
        CUpti_MetricID id = getMetricIdFromName(metricName);
        if (id != std::numeric_limits<CUpti_MetricID>::max())
        {
            collectedMetrics.push_back(std::make_pair(metricName, id));
        }
    }

    return collectedMetrics;
}

void CUDAEngine::getMetricValueCallback(void* userdata, CUpti_CallbackDomain domain, CUpti_CallbackId id, const CUpti_CallbackData* info)
{
    if (id != CUPTI_DRIVER_TRACE_CBID_cuLaunch && id != CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel)
    {
        throw std::runtime_error("Internal CUDA CUPTI error: Unexpected callback id was passed into metric value collection function");
    }

    std::vector<CUDAProfilingMetric>* metrics = reinterpret_cast<std::vector<CUDAProfilingMetric>*>(userdata);

    if (metrics->empty())
    {
        return;
    }
    CUDAProfilingMetric& firstMetric = metrics->at(0);

    if (info->callbackSite == CUPTI_API_ENTER)
    {
        checkCUDAError(cuCtxSynchronize(), "cuCtxSynchronize");
        checkCUDAError(cuptiSetEventCollectionMode(info->context, CUPTI_EVENT_COLLECTION_MODE_KERNEL), "cuptiSetEventCollectionMode");

        for (uint32_t i = 0; i < firstMetric.eventGroupSets->sets[firstMetric.currentSetIndex].numEventGroups; ++i)
        {
            uint32_t profileAll = 1;
            checkCUDAError(cuptiEventGroupSetAttribute(firstMetric.eventGroupSets->sets[firstMetric.currentSetIndex].eventGroups[i],
                CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES, sizeof(profileAll), &profileAll), "cuptiEventGroupSetAttribute");
            checkCUDAError(cuptiEventGroupEnable(firstMetric.eventGroupSets->sets[firstMetric.currentSetIndex].eventGroups[i]),
                "cuptiEventGroupEnable");
        }
    }
    else if (info->callbackSite == CUPTI_API_EXIT)
    {
        checkCUDAError(cuCtxSynchronize(), "cuCtxSynchronize");

        for (uint32_t i = 0; i < firstMetric.eventGroupSets->sets[firstMetric.currentSetIndex].numEventGroups; ++i)
        {
            CUpti_EventGroup group = firstMetric.eventGroupSets->sets[firstMetric.currentSetIndex].eventGroups[i];
            CUpti_EventDomainID groupDomain;
            uint32_t eventCount;
            uint32_t instanceCount;
            uint32_t totalInstanceCount;
            size_t groupDomainSize = sizeof(groupDomain);
            size_t eventCountSize = sizeof(eventCount);
            size_t instanceCountSize = sizeof(instanceCount);
            size_t totalInstanceCountSize = sizeof(totalInstanceCount);

            checkCUDAError(cuptiEventGroupGetAttribute(group, CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID, &groupDomainSize, &groupDomain),
                "cuptiEventGroupGetAttribute");
            checkCUDAError(cuptiEventGroupGetAttribute(group, CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS, &eventCountSize, &eventCount),
                "cuptiEventGroupGetAttribute");
            checkCUDAError(cuptiEventGroupGetAttribute(group, CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT, &instanceCountSize, &instanceCount),
                "cuptiEventGroupGetAttribute");
            checkCUDAError(cuptiDeviceGetEventDomainAttribute(firstMetric.device, groupDomain, CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT,
                &totalInstanceCountSize, &totalInstanceCount), "cuptiDeviceGetEventDomainAttribute");

            std::vector<CUpti_EventID> eventIds(eventCount);
            size_t eventIdsSize = eventCount * sizeof(CUpti_EventID);
            checkCUDAError(cuptiEventGroupGetAttribute(group, CUPTI_EVENT_GROUP_ATTR_EVENTS, &eventIdsSize, eventIds.data()),
                "cuptiEventGroupGetAttribute");

            std::vector<uint64_t> values(instanceCount);
            size_t valuesSize = instanceCount * sizeof(uint64_t);

            for (uint32_t j = 0; j < eventCount; ++j)
            {
                checkCUDAError(cuptiEventGroupReadEvent(group, CUPTI_EVENT_READ_FLAG_NONE, eventIds[j], &valuesSize, values.data()),
                    "cuptiEventGroupReadEvent");

                uint64_t sum = 0;
                for (uint32_t k = 0; k < instanceCount; ++k)
                {
                    sum += values[k];
                }

                const uint64_t normalized = (sum * totalInstanceCount) / instanceCount;
                for (auto& metric : *metrics)
                {
                    for (size_t k = 0; k < metric.eventIds.size(); ++k)
                    {
                        if (metric.eventIds[k] == eventIds[j] && !metric.eventStatuses[k])
                        {
                            metric.eventValues[k] = normalized;
                            metric.eventStatuses[k] = true;
                        }
                    }
                }
            }
        }

        for (uint32_t i = 0; i < firstMetric.eventGroupSets->sets[firstMetric.currentSetIndex].numEventGroups; ++i)
        {
            checkCUDAError(cuptiEventGroupDisable(firstMetric.eventGroupSets->sets[firstMetric.currentSetIndex].eventGroups[i]),
                "cuptiEventGroupDisable");
        }
    }
}

const std::vector<std::string>& CUDAEngine::getDefaultProfilingMetricNames()
{
    static const std::vector<std::string> result
    {
        "achieved_occupancy",
        "branch_efficiency",
        "sm_efficiency",
        "dram_utilization",
        "gld_efficiency",
        "gst_efficiency",
        "dram_read_transactions",
        "dram_write_transactions",
        "shared_utilization",
        "l1_shared_utilization",
        "shared_efficiency",
        "shared_load_transactions",
        "shared_store_transactions",
        "tex_fu_utilization",
        "l2_utilization",
        "alu_fu_utilization",
        "half_precision_fu_utilization",
        "single_precision_fu_utilization",
        "double_precision_fu_utilization",
        "ldst_fu_utilization",
        "special_fu_utilization",
        "inst_executed",
        "inst_fp_16",
        "inst_fp_32",
        "inst_fp_64",
        "inst_integer",
        "inst_inter_thread_communication",
        "inst_misc",
        "inst_replay_overhead"
    };
    return result;
}

#endif // KTT_PROFILING

} // namespace ktt

#endif // KTT_PLATFORM_CUDA
