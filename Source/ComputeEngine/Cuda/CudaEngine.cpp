#ifdef KTT_API_CUDA

#include <ComputeEngine/Cuda/Buffers/CudaDeviceBuffer.h>
#include <ComputeEngine/Cuda/Buffers/CudaHostBuffer.h>
#include <ComputeEngine/Cuda/Buffers/CudaUnifiedBuffer.h>
#include <ComputeEngine/Cuda/CudaDevice.h>
#include <ComputeEngine/Cuda/CudaEngine.h>
#include <ComputeEngine/Cuda/CudaUtility.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/ErrorHandling/KttException.h>
#include <Utility/Logger/Logger.h>
#include <Utility/StlHelpers.h>
#include <Utility/Timer.h>

#ifdef KTT_PROFILING_CUPTI_LEGACY
#include <ComputeEngine/Cuda/CuptiLegacy/CuptiSubscription.h>
#elif KTT_PROFILING_CUPTI
#include <ComputeEngine/Cuda/Cupti/CuptiPass.h>
#endif // KTT_PROFILING_CUPTI

namespace ktt
{

CudaEngine::CudaEngine(const DeviceIndex deviceIndex, const uint32_t queueCount) :
    m_DeviceIndex(deviceIndex),
    m_KernelCache(10)
{
    Logger::LogDebug("Initializing CUDA");
    CheckError(cuInit(0), "cuInit");

    auto devices = CudaDevice::GetAllDevices();

    if (deviceIndex >= static_cast<DeviceIndex>(devices.size()))
    {
        throw KttException("Invalid device index: " + std::to_string(deviceIndex));
    }

    m_Context = std::make_unique<CudaContext>(devices[deviceIndex]);

    for (uint32_t i = 0; i < queueCount; ++i)
    {
        auto stream = std::make_unique<CudaStream>(i);
        m_Streams.push_back(std::move(stream));
    }

    InitializeCompilerOptions();

#if defined(KTT_PROFILING_CUPTI)
    InitializeCupti();
#endif // KTT_PROFILING_CUPTI
}

CudaEngine::CudaEngine(const ComputeApiInitializer& initializer) :
    m_KernelCache(10)
{
    m_Context = std::make_unique<CudaContext>(initializer.GetContext());

    const auto devices = CudaDevice::GetAllDevices();

    for (size_t i = 0; i < devices.size(); ++i)
    {
        if (m_Context->GetDevice() == devices[i].GetDevice())
        {
            m_DeviceIndex = static_cast<DeviceIndex>(i);
            break;
        }
    }

    const auto& streams = initializer.GetQueues();

    for (size_t i = 0; i < streams.size(); ++i)
    {
        auto stream = std::make_unique<CudaStream>(static_cast<QueueId>(i), streams[i]);
        m_Streams.push_back(std::move(stream));
    }

    InitializeCompilerOptions();

#if defined(KTT_PROFILING_CUPTI)
    InitializeCupti();
#endif // KTT_PROFILING_CUPTI
}

ComputeActionId CudaEngine::RunKernelAsync(const KernelComputeData& data, const QueueId queueId)
{
    if (queueId >= static_cast<QueueId>(m_Streams.size()))
    {
        throw KttException("Invalid stream index: " + std::to_string(queueId));
    }

    Timer timer;
    timer.Start();

    auto kernel = LoadKernel(data);
    std::vector<CUdeviceptr*> arguments = GetKernelArguments(data.GetArguments());
    const size_t sharedMemorySize = GetSharedMemorySize(data.GetArguments());

    const auto& stream = *m_Streams[static_cast<size_t>(queueId)];
    auto action = kernel->Launch(stream, data.GetGlobalSize(), data.GetLocalSize(), arguments, sharedMemorySize);
    timer.Stop();

    action->IncreaseOverhead(timer.GetElapsedTime());
    action->SetConfigurationPrefix(data.GetConfigurationPrefix());
    const auto id = action->GetId();
    m_ComputeActions[id] = std::move(action);
    return id;
}

KernelResult CudaEngine::WaitForComputeAction(const ComputeActionId id)
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

KernelResult CudaEngine::RunKernelWithProfiling([[maybe_unused]] const KernelComputeData& data, [[maybe_unused]] const QueueId queueId)
{
    return KernelResult();
}

void CudaEngine::SetProfilingCounters([[maybe_unused]] const std::vector<std::string>& counters)
{
#ifdef KTT_PROFILING_CUPTI_LEGACY
    CuptiInstance::SetEnabledMetrics(counters);
#elif KTT_PROFILING_CUPTI
    m_Profiler->SetCounters(counters);
#else
    throw KttException("Support for kernel profiling is not included in this version of KTT framework");
#endif // KTT_PROFILING_CUPTI_LEGACY
}

bool CudaEngine::IsProfilingSessionActive([[maybe_unused]] const KernelComputeId& id)
{
#if defined(KTT_PROFILING_CUPTI_LEGACY) || defined(KTT_PROFILING_CUPTI)
    return ContainsKey(m_CuptiInstances, id);
#else
    throw KttException("Support for kernel profiling is not included in this version of KTT framework");
#endif // KTT_PROFILING_CUPTI_LEGACY || KTT_PROFILING_CUPTI
}

uint64_t CudaEngine::GetRemainingProfilingRuns([[maybe_unused]] const KernelComputeId& id)
{
#if defined(KTT_PROFILING_CUPTI_LEGACY) || defined(KTT_PROFILING_CUPTI)
    if (!IsProfilingSessionActive(id))
    {
        return 0;
    }

    return m_CuptiInstances[id]->GetRemainingPassCount();
#else
    throw KttException("Support for kernel profiling is not included in this version of KTT framework");
#endif // KTT_PROFILING_CUPTI_LEGACY || KTT_PROFILING_CUPTI
}

bool CudaEngine::HasAccurateRemainingProfilingRuns() const
{
#ifdef KTT_PROFILING_CUPTI_LEGACY
    return true;
#elif KTT_PROFILING_CUPTI
    return false;
#else
    throw KttException("Support for kernel profiling is not included in this version of KTT framework");
#endif // KTT_PROFILING_CUPTI_LEGACY
}

TransferActionId CudaEngine::UploadArgument(KernelArgument& kernelArgument, const QueueId queueId)
{
    return m_Generator.GenerateTransferId();
}

TransferActionId CudaEngine::UpdateArgument(const ArgumentId id, const QueueId queueId, const void* data,
    const size_t dataSize)
{
    return m_Generator.GenerateTransferId();
}

TransferActionId CudaEngine::DownloadArgument(const ArgumentId id, const QueueId queueId, void* destination,
    const size_t dataSize)
{
    return m_Generator.GenerateTransferId();
}

TransferActionId CudaEngine::CopyArgument(const ArgumentId destination, const QueueId queueId, const ArgumentId source,
    const size_t dataSize)
{
    return m_Generator.GenerateTransferId();
}

TransferResult CudaEngine::WaitForTransferAction(const TransferActionId id)
{
    return TransferResult();
}

void CudaEngine::ResizeArgument(const ArgumentId id, const size_t newSize, const bool preserveData)
{

}

void CudaEngine::GetUnifiedMemoryBufferHandle(const ArgumentId id, UnifiedBufferMemory& handle)
{

}

void CudaEngine::AddCustomBuffer(KernelArgument& kernelArgument, ComputeBuffer buffer)
{

}

void CudaEngine::ClearBuffer(const ArgumentId id)
{
    m_Buffers.erase(id);
}

void CudaEngine::ClearBuffers()
{
    m_Buffers.clear();
}

QueueId CudaEngine::GetDefaultQueue() const
{
    return static_cast<QueueId>(0);
}

std::vector<QueueId> CudaEngine::GetAllQueues() const
{
    std::vector<QueueId> result;

    for (const auto& stream : m_Streams)
    {
        result.push_back(stream->GetId());
    }

    return result;
}

void CudaEngine::SynchronizeQueue(const QueueId queueId)
{
    if (static_cast<size_t>(queueId) >= m_Streams.size())
    {
        throw KttException("Invalid CUDA stream index: " + std::to_string(queueId));
    }

    m_Streams[static_cast<size_t>(queueId)]->Synchronize();
}

void CudaEngine::SynchronizeDevice()
{
    for (auto& stream : m_Streams)
    {
        stream->Synchronize();
    }
}

std::vector<PlatformInfo> CudaEngine::GetPlatformInfo() const
{
    int driverVersion;
    CheckError(cuDriverGetVersion(&driverVersion), "cuDriverGetVersion");

    PlatformInfo info(0, "NVIDIA CUDA");
    info.SetVendor("NVIDIA Corporation");
    info.SetVersion(std::to_string(driverVersion));
    info.SetExtensions("N/A");

    return std::vector<PlatformInfo>{info};
}

std::vector<DeviceInfo> CudaEngine::GetDeviceInfo([[maybe_unused]] const PlatformIndex platformIndex) const
{
    std::vector<DeviceInfo> result;

    for (const auto& device : CudaDevice::GetAllDevices())
    {
        result.push_back(device.GetInfo());
    }

    return result;
}

DeviceInfo CudaEngine::GetCurrentDeviceInfo() const
{
    const auto deviceInfos = GetDeviceInfo(0);
    return deviceInfos[static_cast<size_t>(m_DeviceIndex)];
}

void CudaEngine::SetCompilerOptions(const std::string& options)
{
    CudaProgram::SetCompilerOptions(options);
    ClearKernelCache();
}

void CudaEngine::SetGlobalSizeType(const GlobalSizeType type)
{
    CudaKernel::SetGlobalSizeType(type);
}

void CudaEngine::SetAutomaticGlobalSizeCorrection(const bool flag)
{
    CudaKernel::SetGlobalSizeCorrection(flag);
}

void CudaEngine::SetKernelCacheCapacity(const uint64_t capacity)
{
    m_KernelCache.SetMaxSize(static_cast<size_t>(capacity));
}

void CudaEngine::ClearKernelCache()
{
    m_KernelCache.Clear();
}

//uint64_t CUDAEngine::uploadArgument(KernelArgument& kernelArgument)
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
//EventId CUDAEngine::uploadArgumentAsync(KernelArgument& kernelArgument, const QueueId queue)
//{
//    if (queue >= streams.size())
//    {
//        throw std::runtime_error(std::string("Invalid stream index: ") + std::to_string(queue));
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
//    EventId eventId = nextEventId;
//    Logger::logDebug("Uploading buffer for argument " + std::to_string(kernelArgument.getId()) + ", event id: " + std::to_string(eventId));
//    auto buffer = std::make_unique<CUDABuffer>(kernelArgument);
//
//    if (kernelArgument.getMemoryLocation() == ArgumentMemoryLocation::HostZeroCopy)
//    {
//        bufferEvents.insert(std::make_pair(eventId, std::make_pair(std::make_unique<CUDAEvent>(eventId, false),
//            std::make_unique<CUDAEvent>(eventId, false))));
//    }
//    else
//    {
//        auto startEvent = std::make_unique<CUDAEvent>(eventId, true);
//        auto endEvent = std::make_unique<CUDAEvent>(eventId, true);
//        buffer->uploadData(streams.at(queue)->getStream(), kernelArgument.getData(), kernelArgument.getDataSizeInBytes(), startEvent->getEvent(),
//            endEvent->getEvent());
//        bufferEvents.insert(std::make_pair(eventId, std::make_pair(std::move(startEvent), std::move(endEvent))));
//    }
//
//    buffers.insert(std::move(buffer)); // buffer data will be stolen
//    nextEventId++;
//    return eventId;
//}
//
//uint64_t CUDAEngine::updateArgument(const ArgumentId id, const void* data, const size_t dataSizeInBytes)
//{
//    EventId eventId = updateArgumentAsync(id, data, dataSizeInBytes, getDefaultQueue());
//    return getArgumentOperationDuration(eventId);
//}
//
//EventId CUDAEngine::updateArgumentAsync(const ArgumentId id, const void* data, const size_t dataSizeInBytes, const QueueId queue)
//{
//    if (queue >= streams.size())
//    {
//        throw std::runtime_error(std::string("Invalid stream index: ") + std::to_string(queue));
//    }
//
//    CUDABuffer* buffer = findBuffer(id);
//
//    if (buffer == nullptr)
//    {
//        throw std::runtime_error(std::string("Buffer with following id was not found: ") + std::to_string(id));
//    }
//
//    EventId eventId = nextEventId;
//    auto startEvent = std::make_unique<CUDAEvent>(eventId, true);
//    auto endEvent = std::make_unique<CUDAEvent>(eventId, true);
//
//    Logger::getLogger().log(LoggingLevel::Debug, "Updating buffer for argument " + std::to_string(id) + ", event id: " + std::to_string(eventId));
//
//    if (dataSizeInBytes == 0)
//    {
//        buffer->uploadData(streams.at(queue)->getStream(), data, buffer->getBufferSize(), startEvent->getEvent(), endEvent->getEvent());
//    }
//    else
//    {
//        buffer->uploadData(streams.at(queue)->getStream(), data, dataSizeInBytes, startEvent->getEvent(), endEvent->getEvent());
//    }
//
//    bufferEvents.insert(std::make_pair(eventId, std::make_pair(std::move(startEvent), std::move(endEvent))));
//    nextEventId++;
//    return eventId;
//}
//
//uint64_t CUDAEngine::downloadArgument(const ArgumentId id, void* destination, const size_t dataSizeInBytes) const
//{
//    EventId eventId = downloadArgumentAsync(id, destination, dataSizeInBytes, getDefaultQueue());
//    return getArgumentOperationDuration(eventId);
//}
//
//EventId CUDAEngine::downloadArgumentAsync(const ArgumentId id, void* destination, const size_t dataSizeInBytes, const QueueId queue) const
//{
//    if (queue >= streams.size())
//    {
//        throw std::runtime_error(std::string("Invalid stream index: ") + std::to_string(queue));
//    }
//
//    CUDABuffer* buffer = findBuffer(id);
//
//    if (buffer == nullptr)
//    {
//        throw std::runtime_error(std::string("Buffer with following id was not found: ") + std::to_string(id));
//    }
//
//    EventId eventId = nextEventId;
//    auto startEvent = std::make_unique<CUDAEvent>(eventId, true);
//    auto endEvent = std::make_unique<CUDAEvent>(eventId, true);
//
//    Logger::getLogger().log(LoggingLevel::Debug, "Downloading buffer for argument " + std::to_string(id) + ", event id: " + std::to_string(eventId));
//
//    if (dataSizeInBytes == 0)
//    {
//        buffer->downloadData(streams.at(queue)->getStream(), destination, buffer->getBufferSize(), startEvent->getEvent(), endEvent->getEvent());
//    }
//    else
//    {
//        buffer->downloadData(streams.at(queue)->getStream(), destination, dataSizeInBytes, startEvent->getEvent(), endEvent->getEvent());
//    }
//
//    bufferEvents.insert(std::make_pair(eventId, std::make_pair(std::move(startEvent), std::move(endEvent))));
//    nextEventId++;
//    return eventId;
//}
//
//KernelArgument CUDAEngine::downloadArgumentObject(const ArgumentId id, uint64_t* downloadDuration) const
//{
//    CUDABuffer* buffer = findBuffer(id);
//
//    if (buffer == nullptr)
//    {
//        throw std::runtime_error(std::string("Buffer with following id was not found: ") + std::to_string(id));
//    }
//
//    KernelArgument argument(buffer->getKernelArgumentId(), buffer->getBufferSize() / buffer->getElementSize(), buffer->getElementSize(),
//        buffer->getDataType(), buffer->getMemoryLocation(), buffer->getAccessType(), ArgumentUploadType::Vector);
//    
//    EventId eventId = nextEventId;
//    auto startEvent = std::make_unique<CUDAEvent>(eventId, true);
//    auto endEvent = std::make_unique<CUDAEvent>(eventId, true);
//
//    Logger::getLogger().log(LoggingLevel::Debug, "Downloading buffer for argument " + std::to_string(id) + ", event id: " + std::to_string(eventId));
//    buffer->downloadData(streams.at(getDefaultQueue())->getStream(), argument.getData(), argument.getDataSizeInBytes(), startEvent->getEvent(),
//        endEvent->getEvent());
//
//    bufferEvents.insert(std::make_pair(eventId, std::make_pair(std::move(startEvent), std::move(endEvent))));
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
//uint64_t CUDAEngine::copyArgument(const ArgumentId destination, const ArgumentId source, const size_t dataSizeInBytes)
//{
//    EventId eventId = copyArgumentAsync(destination, source, dataSizeInBytes, getDefaultQueue());
//    return getArgumentOperationDuration(eventId);
//}
//
//EventId CUDAEngine::copyArgumentAsync(const ArgumentId destination, const ArgumentId source, const size_t dataSizeInBytes, const QueueId queue)
//{
//    if (queue >= streams.size())
//    {
//        throw std::runtime_error(std::string("Invalid stream index: ") + std::to_string(queue));
//    }
//
//    CUDABuffer* destinationBuffer = findBuffer(destination);
//    CUDABuffer* sourceBuffer = findBuffer(source);
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
//    auto startEvent = std::make_unique<CUDAEvent>(eventId, true);
//    auto endEvent = std::make_unique<CUDAEvent>(eventId, true);
//
//    Logger::getLogger().log(LoggingLevel::Debug, "Copying buffer for argument " + std::to_string(source) + " into buffer for argument "
//        + std::to_string(destination) + ", event id: " + std::to_string(eventId));
//
//    if (dataSizeInBytes == 0)
//    {
//        destinationBuffer->uploadData(streams.at(queue)->getStream(), sourceBuffer, sourceBuffer->getBufferSize(), startEvent->getEvent(),
//            endEvent->getEvent());
//    }
//    else
//    {
//        destinationBuffer->uploadData(streams.at(queue)->getStream(), sourceBuffer, dataSizeInBytes, startEvent->getEvent(), endEvent->getEvent());
//    }
//
//    bufferEvents.insert(std::make_pair(eventId, std::make_pair(std::move(startEvent), std::move(endEvent))));
//    nextEventId++;
//    return eventId;
//}
//
//uint64_t CUDAEngine::persistArgument(KernelArgument& kernelArgument, const bool flag)
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
//        EventId eventId = nextEventId;
//        Logger::logDebug("Uploading persistent buffer for argument " + std::to_string(kernelArgument.getId()) + ", event id: "
//            + std::to_string(eventId));
//        auto buffer = std::make_unique<CUDABuffer>(kernelArgument);
//
//        if (kernelArgument.getMemoryLocation() == ArgumentMemoryLocation::HostZeroCopy)
//        {
//            bufferEvents.insert(std::make_pair(eventId, std::make_pair(std::make_unique<CUDAEvent>(eventId, false),
//                std::make_unique<CUDAEvent>(eventId, false))));
//        }
//        else
//        {
//            auto startEvent = std::make_unique<CUDAEvent>(eventId, true);
//            auto endEvent = std::make_unique<CUDAEvent>(eventId, true);
//            buffer->uploadData(streams.at(getDefaultQueue())->getStream(), kernelArgument.getData(), kernelArgument.getDataSizeInBytes(),
//                startEvent->getEvent(), endEvent->getEvent());
//            bufferEvents.insert(std::make_pair(eventId, std::make_pair(std::move(startEvent), std::move(endEvent))));
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
//uint64_t CUDAEngine::getArgumentOperationDuration(const EventId id) const
//{
//    auto eventPointer = bufferEvents.find(id);
//
//    if (eventPointer == bufferEvents.end())
//    {
//        throw std::runtime_error(std::string("Buffer event with following id does not exist or its result was already retrieved: ")
//            + std::to_string(id));
//    }
//
//    if (!eventPointer->second.first->isValid())
//    {
//        bufferEvents.erase(id);
//        return 0;
//    }
//
//    Logger::getLogger().log(LoggingLevel::Debug, "Performing buffer operation synchronization for event id: " + std::to_string(id));
//
//    // Wait until the second event in pair (the end event) finishes
//    checkCUDAError(cuEventSynchronize(eventPointer->second.second->getEvent()), "cuEventSynchronize");
//    float duration = getEventCommandDuration(eventPointer->second.first->getEvent(), eventPointer->second.second->getEvent());
//    bufferEvents.erase(id);
//
//    return static_cast<uint64_t>(duration);
//}
//
//void CUDAEngine::resizeArgument(const ArgumentId id, const size_t newSize, const bool preserveData)
//{
//    CUDABuffer* buffer = findBuffer(id);
//
//    if (buffer == nullptr)
//    {
//        throw std::runtime_error(std::string("Buffer with following id was not found: ") + std::to_string(id));
//    }
//
//    Logger::getLogger().log(LoggingLevel::Debug, "Resizing buffer for argument " + std::to_string(id));
//    buffer->resize(newSize, preserveData);
//}
//
//void CUDAEngine::getArgumentHandle(const ArgumentId id, BufferMemory& handle)
//{
//    CUDABuffer* buffer = findBuffer(id);
//
//    if (buffer == nullptr)
//    {
//        throw std::runtime_error(std::string("Buffer with following id was not found: ") + std::to_string(id));
//    }
//
//    handle = reinterpret_cast<BufferMemory>(*buffer->getBuffer());
//}
//
//void CUDAEngine::addUserBuffer(UserBuffer buffer, KernelArgument& kernelArgument)
//{
//    if (findBuffer(kernelArgument.getId()) != nullptr)
//    {
//        throw std::runtime_error(std::string("User buffer with the following id already exists: ") + std::to_string(kernelArgument.getId()));
//    }
//
//    auto cudaBuffer = std::make_unique<CUDABuffer>(buffer, kernelArgument);
//    userBuffers.insert(std::move(cudaBuffer));
//}
//
//void CUDAEngine::initializeKernelProfiling(const KernelRuntimeData& kernelData)
//{
//    #ifdef KTT_PROFILING_CUPTI_LEGACY
//    initializeKernelProfiling(kernelData.getName(), kernelData.getSource());
//    #elif KTT_PROFILING_CUPTI
//    initializeKernelProfiling(kernelData.getName(), kernelData.getSource());
//    #else
//    throw std::runtime_error("Support for kernel profiling is not included in this version of KTT framework");
//    #endif // KTT_PROFILING_CUPTI_LEGACY
//}
//
//EventId CUDAEngine::runKernelWithProfiling(const KernelRuntimeData& kernelData, const std::vector<KernelArgument*>& argumentPointers,
//    const QueueId queue)
//{
//    #ifdef KTT_PROFILING_CUPTI_LEGACY
//
//    Timer overheadTimer;
//    overheadTimer.start();
//
//    CUDAKernel* kernel;
//    std::unique_ptr<CUDAKernel> kernelUnique;
//
//    if (kernelCacheFlag)
//    {
//        if (kernelCache.find(std::make_pair(kernelData.getName(), kernelData.getSource())) == kernelCache.end())
//        {
//            if (kernelCache.size() >= kernelCacheCapacity)
//            {
//                clearKernelCache();
//            }
//            std::unique_ptr<CUDAProgram> program = createAndBuildProgram(kernelData.getSource());
//            auto kernel = std::make_unique<CUDAKernel>(program->getPtxSource(), kernelData.getName());
//            kernelCache.insert(std::make_pair(std::make_pair(kernelData.getName(), kernelData.getSource()), std::move(kernel)));
//        }
//        auto cachePointer = kernelCache.find(std::make_pair(kernelData.getName(), kernelData.getSource()));
//        kernel = cachePointer->second.get();
//    }
//    else
//    {
//        std::unique_ptr<CUDAProgram> program = createAndBuildProgram(kernelData.getSource());
//        kernelUnique = std::make_unique<CUDAKernel>(program->getPtxSource(), kernelData.getName());
//        kernel = kernelUnique.get();
//    }
//
//    std::vector<CUdeviceptr*> kernelArguments = getKernelArguments(argumentPointers);
//
//    overheadTimer.stop();
//
//    if (kernelProfilingInstances.find(std::make_pair(kernelData.getName(), kernelData.getSource())) == kernelProfilingInstances.end())
//    {
//        initializeKernelProfiling(kernelData.getName(), kernelData.getSource());
//    }
//
//    auto profilingInstance = kernelProfilingInstances.find(std::make_pair(kernelData.getName(), kernelData.getSource()));
//    EventId id;
//
//    if (!profilingInstance->second->hasValidKernelDuration()) // The first profiling run only captures kernel duration
//    {
//        id = enqueueKernel(*kernel, kernelData.getGlobalSize(), kernelData.getLocalSize(), kernelArguments,
//            getSharedMemorySizeInBytes(argumentPointers, kernelData.getLocalMemoryModifiers()), queue, overheadTimer.getElapsedTime());
//        kernelToEventMap[std::make_pair(kernelData.getName(), kernelData.getSource())].push_back(id);
//
//        Logger::logDebug("Performing kernel synchronization for event id: " + std::to_string(id));
//        auto eventPointer = kernelEvents.find(id);
//        checkCUDAError(cuEventSynchronize(eventPointer->second.second->getEvent()), "cuEventSynchronize");
//        float duration = getEventCommandDuration(eventPointer->second.first->getEvent(), eventPointer->second.second->getEvent());
//        profilingInstance->second->updateState(static_cast<uint64_t>(duration));
//    }
//    else
//    {
//        std::vector<CUPTIProfilingMetric>& metricData = profilingInstance->second->getProfilingMetrics();
//        auto subscription = std::make_unique<CUPTIProfilingSubscription>(metricData);
//
//        id = enqueueKernel(*kernel, kernelData.getGlobalSize(), kernelData.getLocalSize(), kernelArguments,
//            getSharedMemorySizeInBytes(argumentPointers, kernelData.getLocalMemoryModifiers()), queue, overheadTimer.getElapsedTime());
//        kernelToEventMap[std::make_pair(kernelData.getName(), kernelData.getSource())].push_back(id);
//
//        profilingInstance->second->updateState();
//    }
//
//    return id;
//
//    #elif KTT_PROFILING_CUPTI
//
//    Timer overheadTimer;
//    overheadTimer.start();
//
//    CUDAKernel* kernel;
//    std::unique_ptr<CUDAKernel> kernelUnique;
//    auto key = std::make_pair(kernelData.getName(), kernelData.getSource());
//
//    if (kernelCacheFlag)
//    {
//        if (kernelCache.find(key) == kernelCache.end())
//        {
//            if (kernelCache.size() >= kernelCacheCapacity)
//            {
//                clearKernelCache();
//            }
//            std::unique_ptr<CUDAProgram> program = createAndBuildProgram(kernelData.getSource());
//            auto kernel = std::make_unique<CUDAKernel>(program->getPtxSource(), kernelData.getName());
//            kernelCache.insert(std::make_pair(key, std::move(kernel)));
//        }
//        auto cachePointer = kernelCache.find(key);
//        kernel = cachePointer->second.get();
//    }
//    else
//    {
//        std::unique_ptr<CUDAProgram> program = createAndBuildProgram(kernelData.getSource());
//        kernelUnique = std::make_unique<CUDAKernel>(program->getPtxSource(), kernelData.getName());
//        kernel = kernelUnique.get();
//    }
//
//    std::vector<CUdeviceptr*> kernelArguments = getKernelArguments(argumentPointers);
//    overheadTimer.stop();
//
//    if (kernelProfilingInstances.find(key) == kernelProfilingInstances.cend())
//    {
//        initializeKernelProfiling(kernelData.getName(), kernelData.getSource());
//    }
//
//    EventId id;
//    auto profilingInstance = kernelProfilingInstances.find(key);
//
//    if (!profilingInstance->second->hasValidKernelDuration()) // The first profiling run only captures kernel duration
//    {
//        id = enqueueKernel(*kernel, kernelData.getGlobalSize(), kernelData.getLocalSize(), kernelArguments,
//            getSharedMemorySizeInBytes(argumentPointers, kernelData.getLocalMemoryModifiers()), queue, overheadTimer.getElapsedTime());
//        kernelToEventMap[std::make_pair(kernelData.getName(), kernelData.getSource())].push_back(id);
//
//        Logger::logDebug("Performing kernel synchronization for event id: " + std::to_string(id));
//        auto eventPointer = kernelEvents.find(id);
//        checkCUDAError(cuEventSynchronize(eventPointer->second.second->getEvent()), "cuEventSynchronize");
//        float duration = getEventCommandDuration(eventPointer->second.first->getEvent(), eventPointer->second.second->getEvent());
//        profilingInstance->second->setKernelDuration(static_cast<uint64_t>(duration));
//    }
//    else
//    {
//        auto subscription = std::make_unique<CUPTIProfilingPass>(*profilingInstance->second);
//        id = enqueueKernel(*kernel, kernelData.getGlobalSize(), kernelData.getLocalSize(), kernelArguments,
//            getSharedMemorySizeInBytes(argumentPointers, kernelData.getLocalMemoryModifiers()), queue, overheadTimer.getElapsedTime());
//        kernelToEventMap[key].push_back(id);
//
//        auto eventPointer = kernelEvents.find(id);
//        Logger::logDebug(std::string("Performing kernel synchronization for event id: ") + std::to_string(id));
//        checkCUDAError(cuEventSynchronize(eventPointer->second.second->getEvent()), "cuEventSynchronize");
//    }
//
//    return id;
//
//    #else
//    throw std::runtime_error("Support for kernel profiling is not included in this version of KTT framework");
//    #endif // KTT_PROFILING_CUPTI_LEGACY
//}
//
//KernelResult CUDAEngine::getKernelResultWithProfiling(const EventId id, const std::vector<OutputDescriptor>& outputDescriptors)
//{
//    #ifdef KTT_PROFILING_CUPTI_LEGACY
//
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
//    
//    for (const auto eventId : eventIds)
//    {
//        kernelEvents.erase(eventId);
//    }
//
//    kernelToEventMap.erase(kernelKey);
//    return result;
//
//    #elif KTT_PROFILING_CUPTI
//
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
//    result.setComputationDuration(profilingInstance->second->getKernelDuration());
//
//    const CUPTIMetricConfiguration& configuration = profilingInstance->second->getMetricConfiguration();
//    std::vector<CUPTIMetric> metricData = metricInterface->getMetricData(configuration);
//    KernelProfilingData profilingData;
//
//    for (const auto& metric : metricData)
//    {
//        profilingData.addCounter(metric.getCounter());
//    }
//
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
//
//    #else
//    throw std::runtime_error("Support for kernel profiling is not included in this version of KTT framework");
//    #endif // KTT_PROFILING_CUPTI_LEGACY
//}

void CudaEngine::InitializeCompilerOptions()
{
    Logger::LogDebug("Initializing default compiler options");

    int computeCapabilityMajor = 0;
    int computeCapabilityMinor = 0;
    CheckError(cuDeviceGetAttribute(&computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, m_Context->GetDevice()),
        "cuDeviceGetAttribute");
    CheckError(cuDeviceGetAttribute(&computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, m_Context->GetDevice()),
        "cuDeviceGetAttribute");

    const std::string gpuArchitecture = "--gpu-architecture=compute_" + std::to_string(computeCapabilityMajor)
        + std::to_string(computeCapabilityMinor);
    SetCompilerOptions(gpuArchitecture);
}

std::shared_ptr<CudaKernel> CudaEngine::LoadKernel(const KernelComputeData& data)
{
    const auto id = data.GetUniqueIdentifier();

    if (m_KernelCache.GetMaxSize() > 0 && m_KernelCache.Exists(id))
    {
        return m_KernelCache.Get(id)->second;
    }

    auto program = std::make_unique<CudaProgram>(data.GetSource());
    program->Build();
    auto kernel = std::make_shared<CudaKernel>(std::move(program), data.GetName(), m_Generator);

    if (m_KernelCache.GetMaxSize() > 0)
    {
        m_KernelCache.Put(id, kernel);
    }

    return kernel;
}

std::vector<CUdeviceptr*> CudaEngine::GetKernelArguments(const std::vector<const KernelArgument*>& arguments)
{
    std::vector<CUdeviceptr*> result;

    for (const auto* argument : arguments)
    {
        if (argument->GetMemoryType() == ArgumentMemoryType::Local)
        {
            continue;
        }

        CUdeviceptr* deviceArgument = GetKernelArgument(*argument);
        result.push_back(deviceArgument);
    }

    return result;
}

CUdeviceptr* CudaEngine::GetKernelArgument(const KernelArgument& argument)
{
    switch (argument.GetMemoryType())
    {
    case ArgumentMemoryType::Scalar:
        return static_cast<CUdeviceptr*>(const_cast<void*>(argument.GetData()));
    case ArgumentMemoryType::Vector:
    {
        const auto id = argument.GetId();

        if (!ContainsKey(m_Buffers, id))
        {
            throw KttException("Buffer corresponding to kernel argument with id " + std::to_string(id) + " was not found");
        }

        return m_Buffers[id]->GetBuffer();
    }
    case ArgumentMemoryType::Local:
        KttError("Local memory arguments do not have CUdeviceptr representation");
        return nullptr;
    default:
        KttError("Unhandled argument memory type value");
        return nullptr;
    }
}

size_t CudaEngine::GetSharedMemorySize(const std::vector<const KernelArgument*>& arguments) const
{
    size_t result = 0;

    for (const auto* argument : arguments)
    {
        if (argument->GetMemoryType() != ArgumentMemoryType::Local)
        {
            continue;
        }

        result += argument->GetDataSize();
    }

    return result;
}

#if defined(KTT_PROFILING_CUPTI)

void CudaEngine::InitializeCupti()
{
    m_Profiler = std::make_unique<CuptiProfiler>();
    m_MetricInterface = std::make_unique<CuptiMetricInterface>(m_DeviceIndex);
}

#endif // KTT_PROFILING_CUPTI

//#ifdef KTT_PROFILING_CUPTI_LEGACY
//
//void CUDAEngine::initializeKernelProfiling(const std::string& kernelName, const std::string& kernelSource)
//{
//    auto key = std::make_pair(kernelName, kernelSource);
//
//    if (!containsKey(kernelProfilingInstances, key))
//    {
//        kernelProfilingInstances[key] = std::make_unique<CUPTIProfilingInstance>(context->getContext(), context->getDevice(), profilingMetrics);
//        kernelToEventMap[key] = std::vector<EventId>{};
//    }
//}
//
//const std::pair<std::string, std::string>& CUDAEngine::getKernelFromEvent(const EventId id) const
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
//CUpti_MetricID CUDAEngine::getMetricIdFromName(const std::string& metricName)
//{
//    CUpti_MetricID metricId;
//    const CUptiResult result = cuptiMetricGetIdFromName(context->getDevice(), metricName.c_str(), &metricId);
//
//    switch (result)
//    {
//    case CUPTI_SUCCESS:
//        return metricId;
//    case CUPTI_ERROR_INVALID_METRIC_NAME:
//        return std::numeric_limits<CUpti_MetricID>::max();
//    default:
//        checkCUPTIError(result, "cuptiMetricGetIdFromName");
//    }
//
//    return 0;
//}
//
//std::vector<std::pair<std::string, CUpti_MetricID>> CUDAEngine::getProfilingMetricsForCurrentDevice(const std::vector<std::string>& metricNames)
//{
//    std::vector<std::pair<std::string, CUpti_MetricID>> collectedMetrics;
//
//    for (const auto& metricName : metricNames)
//    {
//        CUpti_MetricID id = getMetricIdFromName(metricName);
//        if (id != std::numeric_limits<CUpti_MetricID>::max())
//        {
//            collectedMetrics.push_back(std::make_pair(metricName, id));
//        }
//    }
//
//    return collectedMetrics;
//}
//
//#elif KTT_PROFILING_CUPTI
//
//void CUDAEngine::initializeKernelProfiling(const std::string& kernelName, const std::string& kernelSource)
//{
//    auto key = std::make_pair(kernelName, kernelSource);
//    auto profilingInstance = kernelProfilingInstances.find(key);
//
//    if (profilingInstance == kernelProfilingInstances.end())
//    {
//        if (!kernelProfilingInstances.empty())
//        {
//            throw std::runtime_error("Profiling of multiple kernel instances is not supported for new CUPTI API");
//        }
//
//        const CUPTIMetricConfiguration configuration = metricInterface->createMetricConfiguration(profilingCounters);
//        kernelProfilingInstances.insert(std::make_pair(key, std::make_unique<CUPTIProfilingInstance>(context->getContext(), configuration)));
//        kernelToEventMap.insert(std::make_pair(key, std::vector<EventId>{}));
//    }
//}
//
//const std::pair<std::string, std::string>& CUDAEngine::getKernelFromEvent(const EventId id) const
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
//#endif // KTT_PROFILING_CUPTI

} // namespace ktt

#endif // KTT_API_CUDA
