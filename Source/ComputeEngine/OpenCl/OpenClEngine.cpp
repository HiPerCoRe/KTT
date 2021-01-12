#ifdef KTT_API_OPENCL

#include <ComputeEngine/OpenCl/OpenClEngine.h>
#include <Utility/Logger/Logger.h>
#include <Utility/StlHelpers.h>
#include <Utility/Timer.h>

#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
#include <ComputeEngine/OpenCl/Gpa/GpaProfilingPass.h>
#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY

namespace ktt
{

//OpenClEngine::OpenClEngine(const PlatformIndex platformIndex, const DeviceIndex deviceIndex, const uint32_t queueCount) :
//    m_PlatformIndex(platformIndex),
//    m_DeviceIndex(deviceIndex),
//    m_GlobalSizeType(GlobalSizeType::OpenCL),
//    m_GlobalSizeCorrection(false),
//    m_ComputeIdGenerator(0),
//    m_TransferIdGenerator(0),
//    m_KernelCache(10)
//{
//    #if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
//    Logger::logDebug("Initializing GPA profiling API");
//    gpaInterface = std::make_unique<GPAInterface>();
//    #endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY
//    
//    auto platforms = getOpenCLPlatforms();
//    if (platformIndex >= platforms.size())
//    {
//        throw std::runtime_error(std::string("Invalid platform index: ") + std::to_string(platformIndex));
//    }
//
//    auto devices = getOpenCLDevices(platforms.at(platformIndex));
//    if (deviceIndex >= devices.size())
//    {
//        throw std::runtime_error(std::string("Invalid device index: ") + std::to_string(deviceIndex));
//    }
//
//    cl_device_id device = devices.at(deviceIndex).getId();
//
//    Logger::logDebug("Initializing OpenCL context");
//    context = std::make_unique<OpenCLContext>(platforms.at(platformIndex).getId(), device);
//
//    Logger::logDebug("Initializing OpenCL queues");
//    for (uint32_t i = 0; i < queueCount; i++)
//    {
//        auto commandQueue = std::make_unique<OpenCLCommandQueue>(i, context->getContext(), device);
//        commandQueues.push_back(std::move(commandQueue));
//    }
//
//    initializeProfiler();
//}
//
//OpenClEngine::OpenClEngine(const ComputeApiInitializer& initializer) :
//    compilerOptions(""),
//    globalSizeType(GlobalSizeType::OpenCL),
//    globalSizeCorrection(false),
//    kernelCacheFlag(true),
//    kernelCacheCapacity(10),
//    persistentBufferFlag(true),
//    nextEventId(0)
//{
//    Logger::logDebug("Initializing OpenCL context");
//    context = std::make_unique<OpenCLContext>(initializer.getContext());
//
//    auto platforms = getOpenCLPlatforms();
//
//    for (size_t i = 0; i < platforms.size(); ++i)
//    {
//        if (context->getPlatform() == platforms[i].getId())
//        {
//            platformIndex = static_cast<PlatformIndex>(i);
//            break;
//        }
//    }
//
//    auto devices = getOpenCLDevices(platforms[platformIndex]);
//
//    for (size_t i = 0; i < devices.size(); ++i)
//    {
//        if (context->getDevice() == devices[i].getId())
//        {
//            deviceIndex = static_cast<DeviceIndex>(i);
//            break;
//        }
//    }
//
//    Logger::logDebug("Initializing OpenCL queues");
//    const auto& userQueues = initializer.getQueues();
//
//    for (size_t i = 0; i < userQueues.size(); ++i)
//    {
//        auto commandQueue = std::make_unique<OpenCLCommandQueue>(static_cast<QueueId>(i), context->getContext(), context->getDevice(), userQueues[i]);
//        commandQueues.push_back(std::move(commandQueue));
//    }
//
//    initializeProfiler();
//}
//
//KernelResult OpenCLEngine::runKernel(const KernelRuntimeData& kernelData, const std::vector<KernelArgument*>& argumentPointers,
//    const std::vector<OutputDescriptor>& outputDescriptors)
//{
//    EventId eventId = runKernelAsync(kernelData, argumentPointers, getDefaultQueue());
//    KernelResult result = getKernelResult(eventId, outputDescriptors);
//    return result;
//}
//
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
//void OpenCLEngine::setCompilerOptions(const std::string& options)
//{
//    compilerOptions = options;
//}
//
//void OpenCLEngine::setGlobalSizeType(const GlobalSizeType type)
//{
//    globalSizeType = type;
//}
//
//void OpenCLEngine::setAutomaticGlobalSizeCorrection(const bool flag)
//{
//    globalSizeCorrection = flag;
//}
//
//void OpenCLEngine::setKernelCacheUsage(const bool flag)
//{
//    if (!flag)
//    {
//        clearKernelCache();
//    }
//    kernelCacheFlag = flag;
//}
//
//void OpenCLEngine::setKernelCacheCapacity(const size_t capacity)
//{
//    kernelCacheCapacity = capacity;
//}
//
//void OpenCLEngine::clearKernelCache()
//{
//    kernelCache.clear();
//}
//
//QueueId OpenCLEngine::getDefaultQueue() const
//{
//    return 0;
//}
//
//std::vector<QueueId> OpenCLEngine::getAllQueues() const
//{
//    std::vector<QueueId> result;
//
//    for (size_t i = 0; i < commandQueues.size(); i++)
//    {
//        result.push_back(static_cast<QueueId>(i));
//    }
//
//    return result;
//}
//
//void OpenCLEngine::synchronizeQueue(const QueueId queue)
//{
//    if (queue >= commandQueues.size())
//    {
//        throw std::runtime_error(std::string("Invalid command queue index: ") + std::to_string(queue));
//    }
//
//    checkOpenCLError(clFinish(commandQueues.at(queue)->getQueue()), "clFinish");
//}
//
//void OpenCLEngine::synchronizeDevice()
//{
//    for (auto& commandQueue : commandQueues)
//    {
//        checkOpenCLError(clFinish(commandQueue->getQueue()), "clFinish");
//    }
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
//void OpenCLEngine::setPersistentBufferUsage(const bool flag)
//{
//    persistentBufferFlag = flag;
//}
//
//void OpenCLEngine::clearBuffer(const ArgumentId id)
//{
//    auto iterator = buffers.cbegin();
//
//    while (iterator != buffers.cend())
//    {
//        if (iterator->get()->getKernelArgumentId() == id)
//        {
//            buffers.erase(iterator);
//            return;
//        }
//        else
//        {
//            ++iterator;
//        }
//    }
//}
//
//void OpenCLEngine::clearBuffers()
//{
//    buffers.clear();
//}
//
//void OpenCLEngine::clearBuffers(const ArgumentAccessType accessType)
//{
//    auto iterator = buffers.cbegin();
//
//    while (iterator != buffers.cend())
//    {
//        if (iterator->get()->getOpenclMemoryFlag() == getOpenCLMemoryType(accessType))
//        {
//            iterator = buffers.erase(iterator);
//        }
//        else
//        {
//            ++iterator;
//        }
//    }
//}
//
//void OpenCLEngine::printComputeAPIInfo(std::ostream& outputTarget) const
//{
//    auto platforms = getOpenCLPlatforms();
//
//    for (size_t i = 0; i < platforms.size(); i++)
//    {
//        outputTarget << "Platform " << i << ": " << platforms.at(i).getName() << std::endl;
//        auto devices = getOpenCLDevices(platforms.at(i));
//
//        outputTarget << "Devices for platform " << i << ":" << std::endl;
//        for (size_t j = 0; j < devices.size(); j++)
//        {
//            outputTarget << "Device " << j << ": " << devices.at(j).getName() << std::endl;
//        }
//        outputTarget << std::endl;
//    }
//}
//
//std::vector<PlatformInfo> OpenCLEngine::getPlatformInfo() const
//{
//    std::vector<PlatformInfo> result;
//    auto platforms = getOpenCLPlatforms();
//
//    for (size_t i = 0; i < platforms.size(); i++)
//    {
//        result.push_back(getOpenCLPlatformInfo(static_cast<PlatformIndex>(i)));
//    }
//
//    return result;
//}
//
//std::vector<DeviceInfo> OpenCLEngine::getDeviceInfo(const PlatformIndex platform) const
//{
//    std::vector<DeviceInfo> result;
//    auto platforms = getOpenCLPlatforms();
//    auto devices = getOpenCLDevices(platforms.at(platform));
//
//    for (size_t i = 0; i < devices.size(); i++)
//    {
//        result.push_back(getOpenCLDeviceInfo(platform, static_cast<DeviceIndex>(i)));
//    }
//
//    return result;
//}
//
//DeviceInfo OpenCLEngine::getCurrentDeviceInfo() const
//{
//    return getOpenCLDeviceInfo(platformIndex, deviceIndex);
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
//uint64_t OpenCLEngine::getRemainingKernelProfilingRuns(const std::string& kernelName, const std::string& kernelSource)
//{
//    #if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
//    auto profilingInstance = kernelProfilingInstances.find({kernelName, kernelSource});
//
//    if (profilingInstance == kernelProfilingInstances.end())
//    {
//        return 0;
//    }
//
//    return static_cast<uint64_t>(profilingInstance->second->getRemainingPassCount());
//    #else
//    throw std::runtime_error("Support for kernel profiling is not included in this version of KTT framework");
//    #endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY
//}
//
//bool OpenCLEngine::hasAccurateRemainingKernelProfilingRuns() const
//{
//#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
//    return true;
//#else
//    throw std::runtime_error("Support for kernel profiling is not included in this version of KTT framework");
//#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY
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
//
//std::unique_ptr<OpenCLProgram> OpenCLEngine::createAndBuildProgram(const std::string& source) const
//{
//    auto program = std::make_unique<OpenCLProgram>(source, context->getContext(), std::vector<cl_device_id>{context->getDevice()});
//    program->build(compilerOptions);
//    return program;
//}
//
//void OpenCLEngine::initializeProfiler()
//{
//    #if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
//    Logger::logDebug("Initializing GPA profiling context");
//    gpaProfilingContext = std::make_unique<GPAProfilingContext>(gpaInterface->getFunctionTable(), *commandQueues[getDefaultQueue()].get());
//
//    Logger::logDebug("Initializing default GPA profiling counters");
//    gpaProfilingContext->setCounters(getDefaultGPAProfilingCounters());
//    #endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY
//}
//
//void OpenCLEngine::setKernelArgument(OpenCLKernel& kernel, KernelArgument& argument)
//{
//    if (argument.getUploadType() == ArgumentUploadType::Vector)
//    {
//        if (!loadBufferFromCache(argument.getId(), kernel))
//        {
//            uploadArgument(argument);
//            loadBufferFromCache(argument.getId(), kernel);
//        }
//    }
//    else if (argument.getUploadType() == ArgumentUploadType::Scalar)
//    {
//        kernel.setKernelArgumentScalar(argument.getData(), argument.getElementSizeInBytes());
//    }
//    else
//    {
//        kernel.setKernelArgumentLocal(argument.getElementSizeInBytes() * argument.getNumberOfElements());
//    }
//}
//
//void OpenCLEngine::setKernelArgument(OpenCLKernel& kernel, KernelArgument& argument, const std::vector<LocalMemoryModifier>& modifiers)
//{
//    size_t numberOfElements = argument.getNumberOfElements();
//
//    for (const auto& modifier : modifiers)
//    {
//        if (modifier.getArgument() == argument.getId())
//        {
//            numberOfElements = modifier.getModifiedSize(numberOfElements);
//        }
//    }
//
//    kernel.setKernelArgumentLocal(argument.getElementSizeInBytes() * numberOfElements);
//}
//
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
//OpenCLBuffer* OpenCLEngine::findBuffer(const ArgumentId id) const
//{
//    for (const auto& buffer : userBuffers)
//    {
//        if (buffer->getKernelArgumentId() == id)
//        {
//            return buffer.get();
//        }
//    }
//
//    if (persistentBufferFlag)
//    {
//        for (const auto& buffer : persistentBuffers)
//        {
//            if (buffer->getKernelArgumentId() == id)
//            {
//                return buffer.get();
//            }
//        }
//    }
//
//    for (const auto& buffer : buffers)
//    {
//        if (buffer->getKernelArgumentId() == id)
//        {
//            return buffer.get();
//        }
//    }
//
//    return nullptr;
//}
//
//void OpenCLEngine::setKernelArgumentVector(OpenCLKernel& kernel, const OpenCLBuffer& buffer) const
//{
//    if (buffer.getMemoryLocation() == ArgumentMemoryLocation::Unified)
//    {
//        kernel.setKernelArgumentVectorSVM(buffer.getRawBuffer());
//    }
//    else
//    {
//        cl_mem clBuffer = buffer.getBuffer();
//        kernel.setKernelArgumentVector((void*)&clBuffer);
//    }
//}
//
//bool OpenCLEngine::loadBufferFromCache(const ArgumentId id, OpenCLKernel& kernel) const
//{
//    OpenCLBuffer* buffer = findBuffer(id);
//
//    if (buffer != nullptr)
//    {
//        setKernelArgumentVector(kernel, *buffer);
//        return true;
//    }
//
//    return false;
//}
//
//void OpenCLEngine::checkLocalMemoryModifiers(const std::vector<KernelArgument*>& argumentPointers,
//    const std::vector<LocalMemoryModifier>& modifiers) const
//{
//    for (const auto& modifier : modifiers)
//    {
//        bool modifierArgumentFound = false;
//
//        for (const auto argument : argumentPointers)
//        {
//            if (modifier.getArgument() == argument->getId() && argument->getUploadType() == ArgumentUploadType::Local)
//            {
//                modifierArgumentFound = true;
//            }
//        }
//
//        if (!modifierArgumentFound)
//        {
//            throw std::runtime_error(std::string("No matching local memory argument found for modifier, argument id in modifier: ")
//                + std::to_string(modifier.getArgument()));
//        }
//    }
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
//const std::vector<std::string>& OpenCLEngine::getDefaultGPAProfilingCounters()
//{
//    static const std::vector<std::string> result
//    {
//        "Wavefronts",
//        "VALUInsts",
//        "SALUInsts",
//        "VFetchInsts",
//        "SFetchInsts",
//        "VWriteInsts",
//        "VALUUtilization",
//        "VALUBusy",
//        "SALUBusy",
//        "FetchSize",
//        "WriteSize",
//        "MemUnitBusy",
//        "MemUnitStalled",
//        "WriteUnitStalled"
//    };
//
//    return result;
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
