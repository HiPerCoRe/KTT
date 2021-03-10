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

#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
#include <ComputeEngine/OpenCl/Gpa/GpaPass.h>
#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY

namespace ktt
{

OpenClEngine::OpenClEngine(const PlatformIndex platformIndex, const DeviceIndex deviceIndex, const uint32_t queueCount) :
    m_PlatformIndex(platformIndex),
    m_DeviceIndex(deviceIndex),
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
        auto commandQueue = std::make_unique<OpenClCommandQueue>(i, *m_Context);
        m_Queues.push_back(std::move(commandQueue));
    }

#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
    InitializeGpa();
#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY
}

OpenClEngine::OpenClEngine(const ComputeApiInitializer& initializer) :
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

#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
    InitializeGpa();
#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY
}

ComputeActionId OpenClEngine::RunKernelAsync(const KernelComputeData& data, const QueueId queueId)
{
    if (queueId >= static_cast<QueueId>(m_Queues.size()))
    {
        throw KttException("Invalid queue index: " + std::to_string(queueId));
    }

    Timer timer;
    timer.Start();

    auto kernel = LoadKernel(data);
    SetKernelArguments(*kernel, data.GetArguments());

    const auto& queue = *m_Queues[static_cast<size_t>(queueId)];
    auto action = kernel->Launch(queue, data.GetGlobalSize(), data.GetLocalSize());
    timer.Stop();

    action->IncreaseOverhead(timer.GetElapsedTime());
    action->SetConfigurationPrefix(data.GetConfigurationPrefix());
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
    pass.reset();
    FillProfilingData(id, result);

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

TransferActionId OpenClEngine::UploadArgument(KernelArgument& kernelArgument, const QueueId queueId)
{
    Timer timer;
    timer.Start();

    const auto id = kernelArgument.GetId();
    Logger::LogDebug("Uploading buffer for argument with id " + std::to_string(id));

    if (queueId >= static_cast<QueueId>(m_Queues.size()))
    {
        throw KttException("Invalid queue index: " + std::to_string(queueId));
    }

    if (ContainsKey(m_Buffers, id))
    {
        throw KttException("Buffer for argument with id " + std::to_string(id) + " already exists");
    }

    if (kernelArgument.GetMemoryType() != ArgumentMemoryType::Vector)
    {
        throw KttException("Argument with id " + std::to_string(id) + " is not a vector and cannot be uploaded into buffer");
    }

    auto buffer = CreateBuffer(kernelArgument);
    timer.Stop();

    auto action = buffer->UploadData(*m_Queues[static_cast<size_t>(queueId)], kernelArgument.GetData(),
        kernelArgument.GetDataSize());
    action->IncreaseOverhead(timer.GetElapsedTime());
    const auto actionId = action->GetId();

    m_Buffers[id] = std::move(buffer);
    m_TransferActions[actionId] = std::move(action);

    return actionId;
}

TransferActionId OpenClEngine::UpdateArgument(const ArgumentId id, const QueueId queueId, const void* data,
    const size_t dataSize)
{
    Timer timer;
    timer.Start();

    Logger::LogDebug("Updating buffer for argument with id " + std::to_string(id));

    if (queueId >= static_cast<QueueId>(m_Queues.size()))
    {
        throw KttException("Invalid queue index: " + std::to_string(queueId));
    }

    if (!ContainsKey(m_Buffers, id))
    {
        throw KttException("Buffer for argument with id " + std::to_string(id) + " was not found");
    }

    auto& buffer = *m_Buffers[id];
    size_t actualDataSize = dataSize;

    if (actualDataSize == 0)
    {
        actualDataSize = buffer.GetSize();
    }

    timer.Stop();

    auto action = buffer.UploadData(*m_Queues[static_cast<size_t>(queueId)], data, actualDataSize);
    action->IncreaseOverhead(timer.GetElapsedTime());
    const auto actionId = action->GetId();
    m_TransferActions[actionId] = std::move(action);
    return actionId;
}

TransferActionId OpenClEngine::DownloadArgument(const ArgumentId id, const QueueId queueId, void* destination,
    const size_t dataSize)
{
    Timer timer;
    timer.Start();

    Logger::LogDebug("Downloading buffer for argument with id " + std::to_string(id));

    if (queueId >= static_cast<QueueId>(m_Queues.size()))
    {
        throw KttException("Invalid queue index: " + std::to_string(queueId));
    }

    if (!ContainsKey(m_Buffers, id))
    {
        throw KttException("Buffer for argument with id " + std::to_string(id) + " was not found");
    }

    auto& buffer = *m_Buffers[id];
    size_t actualDataSize = dataSize;

    if (actualDataSize == 0)
    {
        actualDataSize = buffer.GetSize();
    }

    timer.Stop();

    auto action = buffer.DownloadData(*m_Queues[static_cast<size_t>(queueId)], destination, actualDataSize);
    action->IncreaseOverhead(timer.GetElapsedTime());
    const auto actionId = action->GetId();
    m_TransferActions[actionId] = std::move(action);
    return actionId;
}

TransferActionId OpenClEngine::CopyArgument(const ArgumentId destination, const QueueId queueId, const ArgumentId source,
    const size_t dataSize)
{
    Timer timer;
    timer.Start();

    Logger::LogDebug("Copying buffer for argument with id " + std::to_string(source) + " into buffer for argument with id "
        + std::to_string(destination));

    if (queueId >= static_cast<QueueId>(m_Queues.size()))
    {
        throw KttException("Invalid queue index: " + std::to_string(queueId));
    }

    if (!ContainsKey(m_Buffers, destination))
    {
        throw KttException("Copy destination buffer for argument with id " + std::to_string(destination) + " was not found");
    }

    if (!ContainsKey(m_Buffers, source))
    {
        throw KttException("Copy source buffer for argument with id " + std::to_string(source) + " was not found");
    }

    auto& destinationBuffer = *m_Buffers[destination];
    const auto& sourceBuffer = *m_Buffers[source];

    size_t actualDataSize = dataSize;

    if (actualDataSize == 0)
    {
        actualDataSize = sourceBuffer.GetSize();
    }

    timer.Stop();

    auto action = destinationBuffer.CopyData(*m_Queues[static_cast<size_t>(queueId)], sourceBuffer, actualDataSize);
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

void OpenClEngine::ResizeArgument(const ArgumentId id, const size_t newSize, const bool preserveData)
{
    Logger::LogDebug("Resizing buffer for argument with id " + std::to_string(id));

    if (!ContainsKey(m_Buffers, id))
    {
        throw KttException("Buffer for argument with id " + std::to_string(id) + " was not found");
    }

    auto& buffer = *m_Buffers[id];
    buffer.Resize(*m_Queues[static_cast<size_t>(GetDefaultQueue())], newSize, preserveData);
}

void OpenClEngine::GetUnifiedMemoryBufferHandle(const ArgumentId id, UnifiedBufferMemory& handle)
{
    if (!ContainsKey(m_Buffers, id))
    {
        throw KttException("Buffer for argument with id " + std::to_string(id) + " was not found");
    }

    auto& buffer = *m_Buffers[id];

    if (buffer.GetMemoryLocation() != ArgumentMemoryLocation::Unified)
    {
        throw KttException("Buffer for argument with id " + std::to_string(id) + " is not unified memory buffer");
    }

    handle = buffer.GetRawBuffer();
}

void OpenClEngine::AddCustomBuffer(KernelArgument& kernelArgument, ComputeBuffer buffer)
{
    const auto id = kernelArgument.GetId();

    if (ContainsKey(m_Buffers, id))
    {
        throw KttException("Buffer for argument with id " + std::to_string(id) + " already exists");
    }

    auto userBuffer = CreateUserBuffer(kernelArgument, buffer);
    m_Buffers[id] = std::move(userBuffer);
}

void OpenClEngine::ClearBuffer(const ArgumentId id)
{
    m_Buffers.erase(id);
}

void OpenClEngine::ClearBuffers()
{
    m_Buffers.clear();
}

bool OpenClEngine::HasBuffer(const ArgumentId id)
{
    return ContainsKey(m_Buffers, id);
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

void OpenClEngine::SynchronizeQueue(const QueueId queueId)
{
    if (static_cast<size_t>(queueId) >= m_Queues.size())
    {
        throw KttException("Invalid OpenCL command queue index: " + std::to_string(queueId));
    }

    m_Queues[static_cast<size_t>(queueId)]->Synchronize();
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

PlatformInfo OpenClEngine::GetCurrentPlatformInfo() const
{
    const auto platformInfos = GetPlatformInfo();
    return platformInfos[static_cast<size_t>(m_PlatformIndex)];
}

DeviceInfo OpenClEngine::GetCurrentDeviceInfo() const
{
    const auto deviceInfos = GetDeviceInfo(m_PlatformIndex);
    return deviceInfos[static_cast<size_t>(m_DeviceIndex)];
}

void OpenClEngine::SetCompilerOptions(const std::string& options)
{
    OpenClProgram::SetCompilerOptions(options);
    ClearKernelCache();
}

void OpenClEngine::SetGlobalSizeType(const GlobalSizeType type)
{
    OpenClKernel::SetGlobalSizeType(type);
}

void OpenClEngine::SetAutomaticGlobalSizeCorrection(const bool flag)
{
    OpenClKernel::SetGlobalSizeCorrection(flag);
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
    auto kernel = std::make_shared<OpenClKernel>(std::move(program), data.GetName(), m_ComputeIdGenerator);

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
        throw KttException("Buffer corresponding to kernel argument with id " + std::to_string(id) + " was not found");
    }

    kernel.SetArgument(*m_Buffers[id]);
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
