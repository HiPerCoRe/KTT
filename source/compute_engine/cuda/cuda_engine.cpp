#ifdef KTT_PLATFORM_CUDA

#include <stdexcept>
#include <compute_engine/cuda/cuda_engine.h>
#include <utility/ktt_utility.h>
#include <utility/logger.h>
#include <utility/timer.h>

#ifdef KTT_PROFILING_CUPTI_LEGACY
#include <compute_engine/cuda/cupti_legacy/cupti_profiling_subscription.h>
#elif KTT_PROFILING_CUPTI
#include <compute_engine/cuda/cupti/cupti_profiling_pass.h>
#endif // KTT_PROFILING_CUPTI

namespace ktt
{

CUDAEngine::CUDAEngine(const DeviceIndex deviceIndex, const uint32_t queueCount) :
    deviceIndex(deviceIndex),
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

    #ifdef KTT_PROFILING_CUPTI_LEGACY
    Logger::logDebug("Initializing CUPTI profiling metric IDs");
    const std::vector<std::string>& metricNames = getDefaultProfilingMetricNames();
    profilingMetrics = getProfilingMetricsForCurrentDevice(metricNames);
    #elif KTT_PROFILING_CUPTI
    Logger::logDebug("Initializing CUPTI profiler");
    profiler = std::make_unique<CUPTIProfiler>();
    const std::string deviceName = profiler->getDeviceName(deviceIndex);
    metricInterface = std::make_unique<CUPTIMetricInterface>(deviceName);
    profilingCounters = getDefaultProfilingCounters();
    #endif // KTT_PROFILING_CUPTI
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

#ifdef KTT_PROFILING_CUPTI_LEGACY
    kernelToEventMap.clear();
    kernelProfilingInstances.clear();
#elif KTT_PROFILING_CUPTI
    kernelToEventMap.clear();
    kernelProfilingInstances.clear();
#endif // KTT_PROFILING_CUPTI
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

void CUDAEngine::getArgumentHandle(const ArgumentId id, BufferMemory& handle)
{
    CUDABuffer* buffer = findBuffer(id);

    if (buffer == nullptr)
    {
        throw std::runtime_error(std::string("Buffer with following id was not found: ") + std::to_string(id));
    }

    handle = reinterpret_cast<BufferMemory>(*buffer->getBuffer());
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
    #ifdef KTT_PROFILING_CUPTI_LEGACY
    initializeKernelProfiling(kernelData.getName(), kernelData.getSource());
    #elif KTT_PROFILING_CUPTI
    initializeKernelProfiling(kernelData.getName(), kernelData.getSource());
    #else
    throw std::runtime_error("Support for kernel profiling is not included in this version of KTT framework");
    #endif // KTT_PROFILING_CUPTI_LEGACY
}

EventId CUDAEngine::runKernelWithProfiling(const KernelRuntimeData& kernelData, const std::vector<KernelArgument*>& argumentPointers,
    const QueueId queue)
{
    #ifdef KTT_PROFILING_CUPTI_LEGACY

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

    if (kernelProfilingInstances.find(std::make_pair(kernelData.getName(), kernelData.getSource())) == kernelProfilingInstances.end())
    {
        initializeKernelProfiling(kernelData.getName(), kernelData.getSource());
    }

    auto profilingInstance = kernelProfilingInstances.find(std::make_pair(kernelData.getName(), kernelData.getSource()));
    EventId id;

    if (!profilingInstance->second->hasValidKernelDuration()) // The first profiling run only captures kernel duration
    {
        id = enqueueKernel(*kernel, kernelData.getGlobalSize(), kernelData.getLocalSize(), kernelArguments,
            getSharedMemorySizeInBytes(argumentPointers, kernelData.getLocalMemoryModifiers()), queue, overheadTimer.getElapsedTime());
        kernelToEventMap[std::make_pair(kernelData.getName(), kernelData.getSource())].push_back(id);

        Logger::logDebug("Performing kernel synchronization for event id: " + std::to_string(id));
        auto eventPointer = kernelEvents.find(id);
        checkCUDAError(cuEventSynchronize(eventPointer->second.second->getEvent()), "cuEventSynchronize");
        float duration = getEventCommandDuration(eventPointer->second.first->getEvent(), eventPointer->second.second->getEvent());
        profilingInstance->second->updateState(static_cast<uint64_t>(duration));
    }
    else
    {
        std::vector<CUPTIProfilingMetric>& metricData = profilingInstance->second->getProfilingMetrics();
        auto subscription = std::make_unique<CUPTIProfilingSubscription>(metricData);

        id = enqueueKernel(*kernel, kernelData.getGlobalSize(), kernelData.getLocalSize(), kernelArguments,
            getSharedMemorySizeInBytes(argumentPointers, kernelData.getLocalMemoryModifiers()), queue, overheadTimer.getElapsedTime());
        kernelToEventMap[std::make_pair(kernelData.getName(), kernelData.getSource())].push_back(id);

        profilingInstance->second->updateState();
    }

    return id;

    #elif KTT_PROFILING_CUPTI

    Timer overheadTimer;
    overheadTimer.start();

    CUDAKernel* kernel;
    std::unique_ptr<CUDAKernel> kernelUnique;
    auto key = std::make_pair(kernelData.getName(), kernelData.getSource());

    if (kernelCacheFlag)
    {
        if (kernelCache.find(key) == kernelCache.end())
        {
            if (kernelCache.size() >= kernelCacheCapacity)
            {
                clearKernelCache();
            }
            std::unique_ptr<CUDAProgram> program = createAndBuildProgram(kernelData.getSource());
            auto kernel = std::make_unique<CUDAKernel>(program->getPtxSource(), kernelData.getName());
            kernelCache.insert(std::make_pair(key, std::move(kernel)));
        }
        auto cachePointer = kernelCache.find(key);
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

    if (kernelProfilingInstances.find(key) == kernelProfilingInstances.cend())
    {
        initializeKernelProfiling(kernelData.getName(), kernelData.getSource());
    }

    EventId id;
    auto profilingInstance = kernelProfilingInstances.find(key);

    if (!profilingInstance->second->hasValidKernelDuration()) // The first profiling run only captures kernel duration
    {
        id = enqueueKernel(*kernel, kernelData.getGlobalSize(), kernelData.getLocalSize(), kernelArguments,
            getSharedMemorySizeInBytes(argumentPointers, kernelData.getLocalMemoryModifiers()), queue, overheadTimer.getElapsedTime());
        kernelToEventMap[std::make_pair(kernelData.getName(), kernelData.getSource())].push_back(id);

        Logger::logDebug("Performing kernel synchronization for event id: " + std::to_string(id));
        auto eventPointer = kernelEvents.find(id);
        checkCUDAError(cuEventSynchronize(eventPointer->second.second->getEvent()), "cuEventSynchronize");
        float duration = getEventCommandDuration(eventPointer->second.first->getEvent(), eventPointer->second.second->getEvent());
        profilingInstance->second->setKernelDuration(static_cast<uint64_t>(duration));
    }
    else
    {
        auto subscription = std::make_unique<CUPTIProfilingPass>(*profilingInstance->second);
        id = enqueueKernel(*kernel, kernelData.getGlobalSize(), kernelData.getLocalSize(), kernelArguments,
            getSharedMemorySizeInBytes(argumentPointers, kernelData.getLocalMemoryModifiers()), queue, overheadTimer.getElapsedTime());
        kernelToEventMap[key].push_back(id);

        auto eventPointer = kernelEvents.find(id);
        Logger::logDebug(std::string("Performing kernel synchronization for event id: ") + std::to_string(id));
        checkCUDAError(cuEventSynchronize(eventPointer->second.second->getEvent()), "cuEventSynchronize");
    }

    return id;

    #else
    throw std::runtime_error("Support for kernel profiling is not included in this version of KTT framework");
    #endif // KTT_PROFILING_CUPTI_LEGACY
}

uint64_t CUDAEngine::getRemainingKernelProfilingRuns(const std::string& kernelName, const std::string& kernelSource)
{
    #ifdef KTT_PROFILING_CUPTI_LEGACY

    auto key = std::make_pair(kernelName, kernelSource);

    if (!containsKey(kernelProfilingInstances, key))
    {
        return 0;
    }

    return kernelProfilingInstances.find(key)->second->getRemainingKernelRuns();

    #elif KTT_PROFILING_CUPTI

    auto key = std::make_pair(kernelName, kernelSource);
    auto pair = kernelProfilingInstances.find(key);

    if (pair == kernelProfilingInstances.cend())
    {
        return 0;
    }

    return pair->second->getRemainingPasses();

    #else
    throw std::runtime_error("Support for kernel profiling is not included in this version of KTT framework");
    #endif // KTT_PROFILING_CUPTI_LEGACY
}

bool CUDAEngine::hasAccurateRemainingKernelProfilingRuns() const
{
#ifdef KTT_PROFILING_CUPTI_LEGACY
    return true;
#elif KTT_PROFILING_CUPTI
    return false;
#else
    throw std::runtime_error("Support for kernel profiling is not included in this version of KTT framework");
#endif // KTT_PROFILING_CUPTI_LEGACY
}

KernelResult CUDAEngine::getKernelResultWithProfiling(const EventId id, const std::vector<OutputDescriptor>& outputDescriptors)
{
    #ifdef KTT_PROFILING_CUPTI_LEGACY

    KernelResult result = createKernelResult(id);

    for (const auto& descriptor : outputDescriptors)
    {
        downloadArgument(descriptor.getArgumentId(), descriptor.getOutputDestination(), descriptor.getOutputSizeInBytes());
    }

    const std::pair<std::string, std::string>& kernelKey = getKernelFromEvent(id);
    auto profilingInstance = kernelProfilingInstances.find(kernelKey);
    if (profilingInstance == kernelProfilingInstances.end())
    {
        throw std::runtime_error(std::string("No profiling data exists for the following kernel in current configuration: " + kernelKey.first));
    }

    KernelProfilingData profilingData = profilingInstance->second->generateProfilingData();
    result.setProfilingData(profilingData);

    kernelProfilingInstances.erase(kernelKey);
    const std::vector<EventId>& eventIds = kernelToEventMap.find(kernelKey)->second;
    
    for (const auto eventId : eventIds)
    {
        kernelEvents.erase(eventId);
    }

    kernelToEventMap.erase(kernelKey);
    return result;

    #elif KTT_PROFILING_CUPTI

    KernelResult result = createKernelResult(id);

    for (const auto& descriptor : outputDescriptors)
    {
        downloadArgument(descriptor.getArgumentId(), descriptor.getOutputDestination(), descriptor.getOutputSizeInBytes());
    }

    const std::pair<std::string, std::string>& kernelKey = getKernelFromEvent(id);
    auto profilingInstance = kernelProfilingInstances.find(kernelKey);
    if (profilingInstance == kernelProfilingInstances.end())
    {
        throw std::runtime_error(std::string("No profiling data exists for the following kernel in current configuration: " + kernelKey.first));
    }

    result.setComputationDuration(profilingInstance->second->getKernelDuration());

    const CUPTIMetricConfiguration& configuration = profilingInstance->second->getMetricConfiguration();
    std::vector<CUPTIMetric> metricData = metricInterface->getMetricData(configuration);
    KernelProfilingData profilingData;

    for (const auto& metric : metricData)
    {
        profilingData.addCounter(metric.getCounter());
    }

    result.setProfilingData(profilingData);

    kernelProfilingInstances.erase(kernelKey);
    const std::vector<EventId>& eventIds = kernelToEventMap.find(kernelKey)->second;
    for (const auto eventId : eventIds)
    {
        kernelEvents.erase(eventId);
    }
    kernelToEventMap.erase(kernelKey);

    return result;

    #else
    throw std::runtime_error("Support for kernel profiling is not included in this version of KTT framework");
    #endif // KTT_PROFILING_CUPTI_LEGACY
}

void CUDAEngine::setKernelProfilingCounters(const std::vector<std::string>& counterNames)
{
    #ifdef KTT_PROFILING_CUPTI_LEGACY
    profilingMetrics.clear();
    profilingMetrics = getProfilingMetricsForCurrentDevice(counterNames);
    #elif KTT_PROFILING_CUPTI
    profilingCounters = counterNames;
    #else
    throw std::runtime_error("Support for kernel profiling is not included in this version of KTT framework");
    #endif // KTT_PROFILING_CUPTI_LEGACY
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
    auto startEvent = std::make_unique<CUDAEvent>(eventId, kernel.getKernelName(), kernelLaunchOverhead, kernel.getCompilationData());
    auto endEvent = std::make_unique<CUDAEvent>(eventId, kernel.getKernelName(), kernelLaunchOverhead, kernel.getCompilationData());
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
    KernelCompilationData compilationData = eventPointer->second.first->getCompilationData();
    kernelEvents.erase(id);

    KernelResult result(name, static_cast<uint64_t>(duration));
    result.setOverhead(overhead);
    result.setCompilationData(compilationData);

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

#ifdef KTT_PROFILING_CUPTI_LEGACY

void CUDAEngine::initializeKernelProfiling(const std::string& kernelName, const std::string& kernelSource)
{
    auto key = std::make_pair(kernelName, kernelSource);

    if (!containsKey(kernelProfilingInstances, key))
    {
        kernelProfilingInstances[key] = std::make_unique<CUPTIProfilingInstance>(context->getContext(), context->getDevice(), profilingMetrics);
        kernelToEventMap[key] = std::vector<EventId>{};
    }
}

const std::pair<std::string, std::string>& CUDAEngine::getKernelFromEvent(const EventId id) const
{
    for (const auto& entry : kernelToEventMap)
    {
        if (containsElement(entry.second, id))
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
        checkCUPTIError(result, "cuptiMetricGetIdFromName");
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

const std::vector<std::string>& CUDAEngine::getDefaultProfilingMetricNames()
{
    static const std::vector<std::string> result
    {
        "achieved_occupancy",
        "alu_fu_utilization",
        "branch_efficiency",
        "double_precision_fu_utilization",
        "dram_read_transactions",
        "dram_utilization",
        "dram_write_transactions",
        "gld_efficiency",
        "gst_efficiency",
        "half_precision_fu_utilization",
        "inst_executed",
        "inst_fp_16",
        "inst_fp_32",
        "inst_fp_64",
        "inst_integer",
        "inst_inter_thread_communication",
        "inst_misc",
        "inst_replay_overhead",
        "l1_shared_utilization",
        "l2_utilization",
        "ldst_fu_utilization",
        "shared_efficiency",
        "shared_load_transactions",
        "shared_store_transactions",
        "shared_utilization",
        "single_precision_fu_utilization",
        "sm_efficiency",
        "special_fu_utilization",
        "tex_fu_utilization"
    };
    return result;
}

#elif KTT_PROFILING_CUPTI

void CUDAEngine::initializeKernelProfiling(const std::string& kernelName, const std::string& kernelSource)
{
    auto key = std::make_pair(kernelName, kernelSource);
    auto profilingInstance = kernelProfilingInstances.find(key);

    if (profilingInstance == kernelProfilingInstances.end())
    {
        if (!kernelProfilingInstances.empty())
        {
            throw std::runtime_error("Profiling of multiple kernel instances is not supported for new CUPTI API");
        }

        const CUPTIMetricConfiguration configuration = metricInterface->createMetricConfiguration(profilingCounters);
        kernelProfilingInstances.insert(std::make_pair(key, std::make_unique<CUPTIProfilingInstance>(context->getContext(), configuration)));
        kernelToEventMap.insert(std::make_pair(key, std::vector<EventId>{}));
    }
}

const std::pair<std::string, std::string>& CUDAEngine::getKernelFromEvent(const EventId id) const
{
    for (const auto& entry : kernelToEventMap)
    {
        if (containsElement(entry.second, id))
        {
            return entry.first;
        }
    }

    throw std::runtime_error(std::string("Corresponding kernel was not found for event with id: ") + std::to_string(id));
}

const std::vector<std::string>& CUDAEngine::getDefaultProfilingCounters()
{
    static const std::vector<std::string> result
    {
        "dram__sectors_read.sum", // dram_read_transactions
        "dram__sectors_write.sum", // dram_write_transactions
        "dram__throughput.avg.pct_of_peak_sustained_elapsed", // dram_utilization
        "l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_elapsed", // shared_utilization
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum", // shared_load_transactions
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum", // shared_store_transactions
        "lts__t_sectors.avg.pct_of_peak_sustained_elapsed", // l2_utilization
        "sm__warps_active.avg.pct_of_peak_sustained_active", // achieved_occupancy
        "smsp__cycles_active.avg.pct_of_peak_sustained_elapsed", // sm_efficiency
        "smsp__inst_executed_pipe_fp16.sum", // half_precision_fu_utilization
        "smsp__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active", // double_precision_fu_utilization
        "smsp__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active", // ldst_fu_utilization
        "smsp__inst_executed_pipe_tex.avg.pct_of_peak_sustained_active", // tex_fu_utilization
        "smsp__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active", // special_fu_utilization
        "smsp__inst_executed.sum", // inst_executed
        "smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active", // single_precision_fu_utilization
        "smsp__sass_thread_inst_executed_op_fp16_pred_on.sum", // inst_fp_16
        "smsp__sass_thread_inst_executed_op_fp32_pred_on.sum", // inst_fp_32
        "smsp__sass_thread_inst_executed_op_fp64_pred_on.sum", // inst_fp_64
        "smsp__sass_thread_inst_executed_op_integer_pred_on.sum", // inst_integer
        "smsp__sass_thread_inst_executed_op_inter_thread_communication_pred_on.sum", // inst_inter_thread_communication
        "smsp__sass_thread_inst_executed_op_misc_pred_on.sum" // inst_misc
    };

    return result;
}

#endif // KTT_PROFILING_CUPTI

} // namespace ktt

#endif // KTT_PLATFORM_CUDA
