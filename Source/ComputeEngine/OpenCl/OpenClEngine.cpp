#ifdef KTT_API_OPENCL

#include <Api/KttException.h>
#include <ComputeEngine/OpenCl/Buffers/OpenClDeviceBuffer.h>
#include <ComputeEngine/OpenCl/Buffers/OpenClHostBuffer.h>
#include <ComputeEngine/OpenCl/Buffers/OpenClUnifiedBuffer.h>
#include <ComputeEngine/OpenCl/OpenClEngine.h>
#include <ComputeEngine/OpenCl/OpenClPlatform.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/Logger/Logger.h>
#include <Utility/Timer/Timer.h>
#include <Utility/StlHelpers.h>
#include <Utility/StringUtility.h>

#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
#include <ComputeEngine/OpenCl/Gpa/GpaPass.h>
#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY

namespace ktt
{

OpenClEngine::OpenClEngine(const PlatformIndex platformIndex, const DeviceIndex deviceIndex, const uint32_t queueCount) :
    m_Configuration(GlobalSizeType::OpenCL),
    m_PlatformIndex(platformIndex),
    m_DeviceIndex(deviceIndex),
    m_DeviceInfo(0, ""),
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

    for (uint32_t i = 0; i < queueCount; ++i)
    {
        const QueueId id = m_QueueIdGenerator.GenerateId();
        auto commandQueue = std::make_unique<OpenClCommandQueue>(id, *m_Context);
        m_Queues[id] = std::move(commandQueue);
    }

    m_DeviceInfo = GetDeviceInfo(m_PlatformIndex)[m_DeviceIndex];

#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
    InitializeGpa();
#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY
}

OpenClEngine::OpenClEngine(const ComputeApiInitializer& initializer, std::vector<QueueId>& assignedQueueIds) :
    m_Configuration(GlobalSizeType::OpenCL),
    m_DeviceInfo(0, ""),
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

    for (auto& queue : queues)
    {
        const QueueId id = m_QueueIdGenerator.GenerateId();
        auto commandQueue = std::make_unique<OpenClCommandQueue>(id, *m_Context, queue);
        m_Queues[id] = std::move(commandQueue);
        assignedQueueIds.push_back(id);
    }

    m_DeviceInfo = GetDeviceInfo(m_PlatformIndex)[m_DeviceIndex];

#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
    InitializeGpa();
#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY
}

ComputeActionId OpenClEngine::RunKernelAsync(const KernelComputeData& data, const QueueId queueId, const bool powerMeasurementAllowed)
{
    if (!ContainsKey(m_Queues, queueId))
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

    auto kernel = LoadKernel(data);
    SetKernelArguments(*kernel, data.GetArguments());

    const size_t localMemorySize = GetLocalMemorySize(data.GetArguments()) + kernel->GetAttribute(CL_KERNEL_LOCAL_MEM_SIZE);

    if (localMemorySize > m_DeviceInfo.GetLocalMemorySize())
    {
        throw KttException("Local memory usage of " + std::to_string(localMemorySize) + " bytes exceeds current device limit",
            ExceptionReason::DeviceLimitsExceeded);
    }

    const auto& queue = *m_Queues[queueId];
    timer.Stop();

    auto action = kernel->Launch(queue, data.GetGlobalSize(), data.GetLocalSize());

    action->IncreaseOverhead(timer.GetElapsedTime());
    action->IncreaseCompilationOverhead(timer.GetElapsedTime());
    action->SetComputeId(data.GetUniqueIdentifier());
    const auto id = action->GetId();
    m_ComputeActions[id] = std::move(action);
    return id;
}

ComputationResult OpenClEngine::WaitForComputeAction(const ComputeActionId id)
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

void OpenClEngine::ClearData(const KernelComputeId& id)
{
    EraseIf(m_ComputeActions, [&id](const auto& pair)
    {
        return pair.second->GetComputeId() == id;
    });

#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
    m_GpaInstances.erase(id);
#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY
}

void OpenClEngine::ClearKernelData(const std::string& kernelName)
{
    EraseIf(m_ComputeActions, [&kernelName](const auto& pair)
    {
        return StartsWith(pair.second->GetComputeId(), kernelName);
    });

#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
    EraseIf(m_GpaInstances, [&kernelName](const auto& pair)
    {
        return StartsWith(pair.first, kernelName);
    });
#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY
}

ComputationResult OpenClEngine::RunKernelWithProfiling([[maybe_unused]] const KernelComputeData& data,
    [[maybe_unused]] const QueueId queueId)
{
#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
    Timer timer;
    timer.Start();

    const auto id = data.GetUniqueIdentifier();

    if (!IsProfilingSessionActive(id))
    {
        InitializeProfiling(id);
    }

    auto& instance = *m_GpaInstances[id];
    auto pass = std::make_unique<GpaPass>(m_GpaInterface->GetFunctions(), instance);

    timer.Stop();

    const auto actionId = RunKernelAsync(data, queueId);
    auto& action = *m_ComputeActions[actionId];
    action.IncreaseOverhead(timer.GetElapsedTime());

    ComputationResult result = WaitForComputeAction(actionId);

    timer.Start();
    pass.reset();
    FillProfilingData(id, result);
    timer.Stop();

    result.SetDurationData(result.GetDuration(), result.GetOverhead() + timer.GetElapsedTime(), result.GetCompilationOverhead());
    return result;
#else
    throw KttException("Support for kernel profiling is not included in this version of KTT framework");
#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY
}

void OpenClEngine::SetProfilingCounters([[maybe_unused]] const std::vector<std::string>& counters)
{
#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
    m_GpaContext->SetCounters(counters);
#else
    throw KttException("Support for kernel profiling is not included in this version of KTT framework");
#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY
}

bool OpenClEngine::IsProfilingSessionActive([[maybe_unused]] const KernelComputeId& id)
{
#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
    return ContainsKey(m_GpaInstances, id);
#else
    throw KttException("Support for kernel profiling is not included in this version of KTT framework");
#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY
}

uint64_t OpenClEngine::GetRemainingProfilingRuns([[maybe_unused]] const KernelComputeId& id)
{
#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
    if (!IsProfilingSessionActive(id))
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

bool OpenClEngine::SupportsMultiInstanceProfiling() const
{
#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
    return false;
#else
    throw KttException("Support for kernel profiling is not included in this version of KTT framework");
#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY
}

bool OpenClEngine::IsProfilingActive() const
{
    return m_Configuration.IsProfilingActive();
}

void OpenClEngine::SetProfiling(const bool profiling)
{
#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
    m_Configuration.SetProfiling(profiling);
#else
    if (profiling)
        throw KttException("Support for kernel profiling is not included in this version of KTT framework");
#endif
}


TransferActionId OpenClEngine::UploadArgument(KernelArgument& kernelArgument, const QueueId queueId)
{
    Timer timer;
    timer.Start();

    const auto id = kernelArgument.GetId();
    Logger::LogDebug("Uploading buffer for argument with id " + id);

    if (!ContainsKey(m_Queues, queueId))
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

    auto buffer = CreateBuffer(kernelArgument);
    timer.Stop();

    auto action = buffer->UploadData(*m_Queues[queueId], kernelArgument.GetData(),
        kernelArgument.GetDataSize());
    action->IncreaseOverhead(timer.GetElapsedTime());
    const auto actionId = action->GetId();

    m_Buffers[id] = std::move(buffer);
    m_TransferActions[actionId] = std::move(action);

    return actionId;
}

TransferActionId OpenClEngine::UpdateArgument(const ArgumentId& id, const QueueId queueId, const void* data,
    const size_t dataSize)
{
    Timer timer;
    timer.Start();

    Logger::LogDebug("Updating buffer for argument with id " + id);

    if (!ContainsKey(m_Queues, queueId))
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

    timer.Stop();

    auto action = buffer.UploadData(*m_Queues[queueId], data, actualDataSize);
    action->IncreaseOverhead(timer.GetElapsedTime());
    const auto actionId = action->GetId();
    m_TransferActions[actionId] = std::move(action);
    return actionId;
}

TransferActionId OpenClEngine::DownloadArgument(const ArgumentId& id, const QueueId queueId, void* destination,
    const size_t dataSize)
{
    Timer timer;
    timer.Start();

    Logger::LogDebug("Downloading buffer for argument with id " + id);

    if (!ContainsKey(m_Queues, queueId))
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

    timer.Stop();

    auto action = buffer.DownloadData(*m_Queues[queueId], destination, actualDataSize);
    action->IncreaseOverhead(timer.GetElapsedTime());
    const auto actionId = action->GetId();
    m_TransferActions[actionId] = std::move(action);
    return actionId;
}

TransferActionId OpenClEngine::CopyArgument(const ArgumentId& destination, const QueueId queueId, const ArgumentId& source,
    const size_t dataSize)
{
    Timer timer;
    timer.Start();

    Logger::LogDebug("Copying buffer for argument with id " + source + " into buffer for argument with id "
        + destination);

    if (!ContainsKey(m_Queues, queueId))
    {
        throw KttException("Invalid queue index: " + std::to_string(queueId));
    }

    if (!ContainsKey(m_Buffers, destination))
    {
        throw KttException("Copy destination buffer for argument with id " + destination + " was not found");
    }

    if (!ContainsKey(m_Buffers, source))
    {
        throw KttException("Copy source buffer for argument with id " + source + " was not found");
    }

    auto& destinationBuffer = *m_Buffers[destination];
    const auto& sourceBuffer = *m_Buffers[source];

    size_t actualDataSize = dataSize;

    if (actualDataSize == 0)
    {
        actualDataSize = sourceBuffer.GetSize();
    }

    timer.Stop();

    auto action = destinationBuffer.CopyData(*m_Queues[queueId], sourceBuffer, actualDataSize);
    action->IncreaseOverhead(timer.GetElapsedTime());
    const auto actionId = action->GetId();
    m_TransferActions[actionId] = std::move(action);
    return actionId;
}

TransferResult OpenClEngine::WaitForTransferAction(const TransferActionId id)
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

void OpenClEngine::ResizeArgument(const ArgumentId& id, const size_t newSize, const bool preserveData)
{
    Logger::LogDebug("Resizing buffer for argument with id " + id);

    if (!ContainsKey(m_Buffers, id))
    {
        throw KttException("Buffer for argument with id " + id + " was not found");
    }

    auto& buffer = *m_Buffers[id];
    buffer.Resize(*m_Queues[GetDefaultQueue()], newSize, preserveData);
}

void OpenClEngine::GetUnifiedMemoryBufferHandle(const ArgumentId& id, UnifiedBufferMemory& handle)
{
    if (!ContainsKey(m_Buffers, id))
    {
        throw KttException("Buffer for argument with id " + id + " was not found");
    }

    auto& buffer = *m_Buffers[id];

    if (buffer.GetMemoryLocation() != ArgumentMemoryLocation::Unified)
    {
        throw KttException("Buffer for argument with id " + id + " is not unified memory buffer");
    }

    handle = buffer.GetRawBuffer();
}

void OpenClEngine::AddCustomBuffer(KernelArgument& kernelArgument, ComputeBuffer buffer)
{
    const auto id = kernelArgument.GetId();

    if (ContainsKey(m_Buffers, id))
    {
        throw KttException("Buffer for argument with id " + id + " already exists");
    }

    auto userBuffer = CreateUserBuffer(kernelArgument, buffer);
    m_Buffers[id] = std::move(userBuffer);
}

void OpenClEngine::ClearBuffer(const ArgumentId& id)
{
    m_Buffers.erase(id);
}

void OpenClEngine::ClearBuffers()
{
    m_Buffers.clear();
}

bool OpenClEngine::HasBuffer(const ArgumentId& id)
{
    return ContainsKey(m_Buffers, id);
}

QueueId OpenClEngine::AddComputeQueue(ComputeQueue queue)
{
    if (!m_Context->IsUserOwned())
    {
        throw KttException("New OpenCL queues cannot be added to tuner which was not created with compute API initializer");
    }

    for (const auto& commandQueue : m_Queues)
    {
        if (commandQueue.second->GetQueue() == static_cast<cl_command_queue>(queue))
        {
            throw KttException("The provided OpenCL queue already exists inside the tuner under id: "
                + std::to_string(commandQueue.first));
        }
    }

    const QueueId id = m_QueueIdGenerator.GenerateId();
    auto commandQueue = std::make_unique<OpenClCommandQueue>(id, *m_Context, queue);
    m_Queues[id] = std::move(commandQueue);
    return id;
}

void OpenClEngine::RemoveComputeQueue(const QueueId id)
{
    if (!m_Context->IsUserOwned())
    {
        throw KttException("OpenCL command queues cannot be removed from tuner which was not created with compute API initializer");
    }

    if (!ContainsKey(m_Queues, id))
    {
        throw KttException("Invalid queue index: " + std::to_string(id));
    }

    m_Queues.erase(id);
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
        result.push_back(queue.first);
    }

    return result;
}

void OpenClEngine::SynchronizeQueue(const QueueId queueId)
{
    if (!ContainsKey(m_Queues, queueId))
    {
        throw KttException("Invalid OpenCL command queue index: " + std::to_string(queueId));
    }

    m_Queues[queueId]->Synchronize();
    ClearQueueActions(queueId);
}

void OpenClEngine::SynchronizeQueues()
{
    for (auto& queue : m_Queues)
    {
        queue.second->Synchronize();
        ClearQueueActions(queue.first);
    }
}

void OpenClEngine::SynchronizeDevice()
{
    SynchronizeQueues();
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

PlatformInfo OpenClEngine::GetCurrentPlatformInfo() const
{
    const auto platformInfos = GetPlatformInfo();
    return platformInfos[static_cast<size_t>(m_PlatformIndex)];
}

DeviceInfo OpenClEngine::GetCurrentDeviceInfo() const
{
    return m_DeviceInfo;
}

ComputeApi OpenClEngine::GetComputeApi() const
{
    return ComputeApi::OpenCL;
}

GlobalSizeType OpenClEngine::GetGlobalSizeType() const
{
    return m_Configuration.GetGlobalSizeType();
}

void OpenClEngine::SetCompilerOptions(const std::string& options, [[maybe_unused]] const bool overrideDefault)
{
    m_Configuration.SetCompilerOptions(options);
    ClearKernelCache();
}

void OpenClEngine::SetGlobalSizeType(const GlobalSizeType type)
{
    m_Configuration.SetGlobalSizeType(type);
}

void OpenClEngine::SetAutomaticGlobalSizeCorrection(const bool flag)
{
    m_Configuration.SetGlobalSizeCorrection(flag);
}

void OpenClEngine::SetKernelCacheCapacity(const uint64_t capacity)
{
    m_KernelCache.SetMaxSize(static_cast<size_t>(capacity));
}

void OpenClEngine::ClearKernelCache()
{
    m_KernelCache.Clear();
}

void OpenClEngine::EnsureThreadContext()
{}

std::shared_ptr<OpenClKernel> OpenClEngine::LoadKernel(const KernelComputeData& data)
{
    const auto id = data.GetUniqueIdentifier();

    if (m_KernelCache.GetMaxSize() > 0 && m_KernelCache.Exists(id))
    {
        return m_KernelCache.Get(id)->second;
    }

    auto program = std::make_unique<OpenClProgram>(*m_Context, data.GetSource());
    program->Build(m_Configuration.GetCompilerOptions());
    auto kernel = std::make_shared<OpenClKernel>(std::move(program), data.GetName(), m_ComputeIdGenerator, m_Configuration);

    if (m_KernelCache.GetMaxSize() > 0)
    {
        m_KernelCache.Put(id, kernel);
    }

    return kernel;
}

void OpenClEngine::SetKernelArguments(OpenClKernel& kernel, const std::vector<KernelArgument*> arguments)
{
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
        throw KttException("Buffer corresponding to kernel argument with id " + id + " was not found");
    }

    kernel.SetArgument(*m_Buffers[id]);
}

size_t OpenClEngine::GetLocalMemorySize(const std::vector<KernelArgument*>& arguments) const
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

std::unique_ptr<OpenClBuffer> OpenClEngine::CreateBuffer(KernelArgument& argument)
{
    std::unique_ptr<OpenClBuffer> buffer;

    switch (argument.GetMemoryLocation())
    {
    case ArgumentMemoryLocation::Undefined:
        KttError("Buffer cannot be created for arguments with undefined memory location");
        break;
    case ArgumentMemoryLocation::Device:
        buffer = std::make_unique<OpenClDeviceBuffer>(argument, m_TransferIdGenerator, *m_Context);
        break;
    case ArgumentMemoryLocation::Host:
    case ArgumentMemoryLocation::HostZeroCopy:
        buffer = std::make_unique<OpenClHostBuffer>(argument, m_TransferIdGenerator, *m_Context);
        break;
    case ArgumentMemoryLocation::Unified:
#ifdef CL_VERSION_2_0
        buffer = std::make_unique<OpenClUnifiedBuffer>(argument, m_TransferIdGenerator, *m_Context);
        break;
#else
        throw KttException("Unified memory buffers are not supported on this platform");
#endif // CL_VERSION_2_0
    default:
        KttError("Unhandled argument memory location value");
        break;
    }

    return buffer;
}

std::unique_ptr<OpenClBuffer> OpenClEngine::CreateUserBuffer(KernelArgument& argument, ComputeBuffer buffer)
{
    std::unique_ptr<OpenClBuffer> userBuffer;

    switch (argument.GetMemoryLocation())
    {
    case ArgumentMemoryLocation::Undefined:
        KttError("Buffer cannot be created for arguments with undefined memory location");
        break;
    case ArgumentMemoryLocation::Device:
        userBuffer = std::make_unique<OpenClDeviceBuffer>(argument, m_TransferIdGenerator, buffer);
        break;
    case ArgumentMemoryLocation::Host:
    case ArgumentMemoryLocation::HostZeroCopy:
        userBuffer = std::make_unique<OpenClHostBuffer>(argument, m_TransferIdGenerator, buffer);
        break;
    case ArgumentMemoryLocation::Unified:
#ifdef CL_VERSION_2_0
        userBuffer = std::make_unique<OpenClUnifiedBuffer>(argument, m_TransferIdGenerator, buffer);
        break;
#else
        throw KttException("Unified memory buffers are not supported on this platform");
#endif // CL_VERSION_2_0
    default:
        KttError("Unhandled argument memory location value");
        break;
    }

    return userBuffer;
}

void OpenClEngine::ClearQueueActions(const QueueId id)
{
    EraseIf(m_ComputeActions, [id](const auto& pair)
    {
        return pair.second->GetQueueId() == id;
    });

    EraseIf(m_TransferActions, [id](const auto& pair)
    {
        return pair.second->GetQueueId() == id;
    });
}

#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)

void OpenClEngine::InitializeGpa()
{
    m_GpaInterface = std::make_unique<GpaInterface>();
    m_GpaContext = std::make_unique<GpaContext>(m_GpaInterface->GetFunctions(), *m_Queues[GetDefaultQueue()]);
}

void OpenClEngine::InitializeProfiling(const KernelComputeId& id)
{
    KttAssert(!IsProfilingSessionActive(id), "Attempting to initialize profiling for compute id with active profiling session");
    m_GpaInstances[id] = std::make_unique<GpaInstance>(m_GpaInterface->GetFunctions(), *m_GpaContext);
}

void OpenClEngine::FillProfilingData(const KernelComputeId& id, ComputationResult& result)
{
    KttAssert(IsProfilingSessionActive(id), "Attempting to retrieve profiling data for kernel without active profiling session");

    auto& instance = *m_GpaInstances[id];
    auto profilingData = instance.GenerateProfilingData();

    if (profilingData->IsValid())
    {
        m_GpaInstances.erase(id);
    }

    result.SetProfilingData(std::move(profilingData));
}

#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY

} // namespace ktt

#endif // KTT_PLATFORM_OPENCL
