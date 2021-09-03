#ifdef KTT_API_CUDA

#include <Api/KttException.h>
#include <ComputeEngine/Cuda/Buffers/CudaDeviceBuffer.h>
#include <ComputeEngine/Cuda/Buffers/CudaHostBuffer.h>
#include <ComputeEngine/Cuda/Buffers/CudaUnifiedBuffer.h>
#include <ComputeEngine/Cuda/CudaDevice.h>
#include <ComputeEngine/Cuda/CudaEngine.h>
#include <ComputeEngine/Cuda/CudaUtility.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/Logger/Logger.h>
#include <Utility/Timer/Timer.h>
#include <Utility/StlHelpers.h>
#include <Utility/StringUtility.h>

#ifdef KTT_PROFILING_CUPTI_LEGACY
#include <ComputeEngine/Cuda/CuptiLegacy/CuptiSubscription.h>
#elif KTT_PROFILING_CUPTI
#include <ComputeEngine/Cuda/Cupti/CuptiPass.h>
#endif // KTT_PROFILING_CUPTI

namespace ktt
{

CudaEngine::CudaEngine(const DeviceIndex deviceIndex, const uint32_t queueCount) :
    m_Configuration(GlobalSizeType::CUDA),
    m_DeviceIndex(deviceIndex),
    m_DeviceInfo(0, ""),
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
    m_DeviceInfo = GetDeviceInfo(0)[m_DeviceIndex];

#if defined(KTT_PROFILING_CUPTI)
    InitializeCupti();
#endif // KTT_PROFILING_CUPTI
}

CudaEngine::CudaEngine(const ComputeApiInitializer& initializer) :
    m_Configuration(GlobalSizeType::CUDA),
    m_DeviceInfo(0, ""),
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
    m_DeviceInfo = GetDeviceInfo(0)[m_DeviceIndex];

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

    const uint64_t localSize = static_cast<uint64_t>(data.GetLocalSize().GetTotalSize());

    if (localSize > m_DeviceInfo.GetMaxWorkGroupSize())
    {
        throw KttException("Block size of " + std::to_string(localSize) + " exceeds current device limit",
            ExceptionReason::DeviceLimitsExceeded);
    }

    Timer timer;
    timer.Start();

    auto kernel = LoadKernel(data);
    std::vector<CUdeviceptr*> arguments = GetKernelArguments(data.GetArguments());
    const size_t sharedMemorySize = GetSharedMemorySize(data.GetArguments());

    const auto& stream = *m_Streams[static_cast<size_t>(queueId)];
    timer.Stop();

    auto action = kernel->Launch(stream, data.GetGlobalSize(), data.GetLocalSize(), arguments, sharedMemorySize);

    action->IncreaseOverhead(timer.GetElapsedTime());
    action->SetComputeId(data.GetUniqueIdentifier());
    const auto id = action->GetId();
    m_ComputeActions[id] = std::move(action);
    return id;
}

ComputationResult CudaEngine::WaitForComputeAction(const ComputeActionId id)
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

void CudaEngine::ClearData(const KernelComputeId& id)
{
    EraseIf(m_ComputeActions, [&id](const auto& pair)
    {
        return pair.second->GetComputeId() == id;
    });

#if defined(KTT_PROFILING_CUPTI_LEGACY) || defined(KTT_PROFILING_CUPTI)
    m_CuptiInstances.erase(id);
#endif // KTT_PROFILING_CUPTI_LEGACY || KTT_PROFILING_CUPTI
}

void CudaEngine::ClearKernelData(const std::string& kernelName)
{
    EraseIf(m_ComputeActions, [&kernelName](const auto& pair)
    {
        return StartsWith(pair.second->GetComputeId(), kernelName);
    });

#if defined(KTT_PROFILING_CUPTI_LEGACY) || defined(KTT_PROFILING_CUPTI)
    EraseIf(m_CuptiInstances, [&kernelName](const auto& pair)
    {
        return StartsWith(pair.first, kernelName);
    });
#endif // KTT_PROFILING_CUPTI_LEGACY || KTT_PROFILING_CUPTI
}

ComputationResult CudaEngine::RunKernelWithProfiling([[maybe_unused]] const KernelComputeData& data, [[maybe_unused]] const QueueId queueId)
{
#ifdef KTT_PROFILING_CUPTI_LEGACY

    Timer timer;
    timer.Start();

    const auto id = data.GetUniqueIdentifier();

    if (!IsProfilingSessionActive(id))
    {
        InitializeProfiling(id);
    }

    auto& instance = *m_CuptiInstances[id];
    std::unique_ptr<CuptiSubscription> subscription;

    if (instance.HasValidKernelDuration())
    {
        subscription = std::make_unique<CuptiSubscription>(instance);
    }

    timer.Stop();

    const auto actionId = RunKernelAsync(data, queueId);
    auto& action = *m_ComputeActions[actionId];
    action.IncreaseOverhead(timer.GetElapsedTime());
    ComputationResult result = WaitForComputeAction(actionId);

    if (!instance.HasValidKernelDuration())
    {
        instance.SetKernelDuration(result.GetDuration());
    }

    timer.Start();
    subscription.reset();
    FillProfilingData(id, result);
    timer.Stop();

    result.SetDurationData(result.GetDuration(), result.GetOverhead() + timer.GetElapsedTime());
    return result;

#elif KTT_PROFILING_CUPTI

    Timer timer;
    timer.Start();

    const auto id = data.GetUniqueIdentifier();

    if (!IsProfilingSessionActive(id))
    {
        InitializeProfiling(id);
    }

    auto& instance = *m_CuptiInstances[id];
    std::unique_ptr<CuptiPass> pass;

    if (instance.HasValidKernelDuration())
    {
        pass = std::make_unique<CuptiPass>(instance);
    }

    timer.Stop();

    const auto actionId = RunKernelAsync(data, queueId);
    auto& action = *m_ComputeActions[actionId];
    action.IncreaseOverhead(timer.GetElapsedTime()); 
    ComputationResult result = WaitForComputeAction(actionId);
    
    if (!instance.HasValidKernelDuration())
    {
        instance.SetKernelDuration(result.GetDuration());
    }

    timer.Start();
    pass.reset();
    FillProfilingData(id, result);
    timer.Stop();

    result.SetDurationData(result.GetDuration(), result.GetOverhead() + timer.GetElapsedTime());
    return result;

#else
    throw KttException("Support for kernel profiling is not included in this version of KTT framework");
#endif // KTT_PROFILING_CUPTI_LEGACY || KTT_PROFILING_CUPTI
}

void CudaEngine::SetProfilingCounters([[maybe_unused]] const std::vector<std::string>& counters)
{
#ifdef KTT_PROFILING_CUPTI_LEGACY
    CuptiInstance::SetEnabledMetrics(counters);
#elif KTT_PROFILING_CUPTI
    m_MetricInterface->SetMetrics(counters);
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
#ifdef KTT_PROFILING_CUPTI_LEGACY

    if (!IsProfilingSessionActive(id))
    {
        return 0;
    }

    const auto& instance = *m_CuptiInstances[id];
    uint64_t passCount = instance.GetRemainingPassCount();

    if (!instance.HasValidKernelDuration())
    {
        passCount += 1;
    }

    return passCount;

#elif KTT_PROFILING_CUPTI

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

bool CudaEngine::SupportsMultiInstanceProfiling() const
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
    Timer timer;
    timer.Start();

    const auto id = kernelArgument.GetId();
    Logger::LogDebug("Uploading buffer for argument with id " + std::to_string(id));

    if (queueId >= static_cast<QueueId>(m_Streams.size()))
    {
        throw KttException("Invalid stream index: " + std::to_string(queueId));
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

    auto action = buffer->UploadData(*m_Streams[static_cast<size_t>(queueId)], kernelArgument.GetData(),
        kernelArgument.GetDataSize());
    action->IncreaseOverhead(timer.GetElapsedTime());
    const auto actionId = action->GetId();

    m_Buffers[id] = std::move(buffer);
    m_TransferActions[actionId] = std::move(action);

    return actionId;
}

TransferActionId CudaEngine::UpdateArgument(const ArgumentId id, const QueueId queueId, const void* data,
    const size_t dataSize)
{
    Timer timer;
    timer.Start();

    Logger::LogDebug("Updating buffer for argument with id " + std::to_string(id));

    if (queueId >= static_cast<QueueId>(m_Streams.size()))
    {
        throw KttException("Invalid stream index: " + std::to_string(queueId));
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

    auto action = buffer.UploadData(*m_Streams[static_cast<size_t>(queueId)], data, actualDataSize);
    action->IncreaseOverhead(timer.GetElapsedTime());
    const auto actionId = action->GetId();
    m_TransferActions[actionId] = std::move(action);
    return actionId;
}

TransferActionId CudaEngine::DownloadArgument(const ArgumentId id, const QueueId queueId, void* destination,
    const size_t dataSize)
{
    Timer timer;
    timer.Start();

    Logger::LogDebug("Downloading buffer for argument with id " + std::to_string(id));

    if (queueId >= static_cast<QueueId>(m_Streams.size()))
    {
        throw KttException("Invalid stream index: " + std::to_string(queueId));
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

    auto action = buffer.DownloadData(*m_Streams[static_cast<size_t>(queueId)], destination, actualDataSize);
    action->IncreaseOverhead(timer.GetElapsedTime());
    const auto actionId = action->GetId();
    m_TransferActions[actionId] = std::move(action);
    return actionId;
}

TransferActionId CudaEngine::CopyArgument(const ArgumentId destination, const QueueId queueId, const ArgumentId source,
    const size_t dataSize)
{
    Timer timer;
    timer.Start();

    Logger::LogDebug("Copying buffer for argument with id " + std::to_string(source) + " into buffer for argument with id "
        + std::to_string(destination));

    if (queueId >= static_cast<QueueId>(m_Streams.size()))
    {
        throw KttException("Invalid stream index: " + std::to_string(queueId));
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

    auto action = destinationBuffer.CopyData(*m_Streams[static_cast<size_t>(queueId)], sourceBuffer, actualDataSize);
    action->IncreaseOverhead(timer.GetElapsedTime());
    const auto actionId = action->GetId();
    m_TransferActions[actionId] = std::move(action);
    return actionId;
}

TransferResult CudaEngine::WaitForTransferAction(const TransferActionId id)
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

void CudaEngine::ResizeArgument(const ArgumentId id, const size_t newSize, const bool preserveData)
{
    Logger::LogDebug("Resizing buffer for argument with id " + std::to_string(id));

    if (!ContainsKey(m_Buffers, id))
    {
        throw KttException("Buffer for argument with id " + std::to_string(id) + " was not found");
    }

    auto& buffer = *m_Buffers[id];
    buffer.Resize(newSize, preserveData);
}

void CudaEngine::GetUnifiedMemoryBufferHandle(const ArgumentId id, UnifiedBufferMemory& handle)
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

    handle = reinterpret_cast<UnifiedBufferMemory>(*buffer.GetBuffer());
}

void CudaEngine::AddCustomBuffer(KernelArgument& kernelArgument, ComputeBuffer buffer)
{
    const auto id = kernelArgument.GetId();

    if (ContainsKey(m_Buffers, id))
    {
        throw KttException("Buffer for argument with id " + std::to_string(id) + " already exists");
    }

    auto userBuffer = CreateUserBuffer(kernelArgument, buffer);
    m_Buffers[id] = std::move(userBuffer);
}

void CudaEngine::ClearBuffer(const ArgumentId id)
{
    m_Buffers.erase(id);
}

void CudaEngine::ClearBuffers()
{
    m_Buffers.clear();
}

bool CudaEngine::HasBuffer(const ArgumentId id)
{
    return ContainsKey(m_Buffers, id);
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

PlatformInfo CudaEngine::GetCurrentPlatformInfo() const
{
    return GetPlatformInfo()[0];
}

DeviceInfo CudaEngine::GetCurrentDeviceInfo() const
{
    return m_DeviceInfo;
}

ComputeApi CudaEngine::GetComputeApi() const
{
    return ComputeApi::CUDA;
}

GlobalSizeType CudaEngine::GetGlobalSizeType() const
{
    return m_Configuration.GetGlobalSizeType();
}

void CudaEngine::SetCompilerOptions(const std::string& options)
{
    m_Configuration.SetCompilerOptions(options);
    ClearKernelCache();
}

void CudaEngine::SetGlobalSizeType(const GlobalSizeType type)
{
    m_Configuration.SetGlobalSizeType(type);
}

void CudaEngine::SetAutomaticGlobalSizeCorrection(const bool flag)
{
    m_Configuration.SetGlobalSizeCorrection(flag);
}

void CudaEngine::SetKernelCacheCapacity(const uint64_t capacity)
{
    m_KernelCache.SetMaxSize(static_cast<size_t>(capacity));
}

void CudaEngine::ClearKernelCache()
{
    m_KernelCache.Clear();
}

void CudaEngine::EnsureThreadContext()
{
    m_Context->EnsureThreadContext();
}

std::shared_ptr<CudaKernel> CudaEngine::LoadKernel(const KernelComputeData& data)
{
    const auto id = data.GetUniqueIdentifier();

    if (m_KernelCache.GetMaxSize() > 0 && m_KernelCache.Exists(id))
    {
        return m_KernelCache.Get(id)->second;
    }

    const auto symbolArguments = KernelArgument::GetArgumentsWithMemoryType(data.GetArguments(), ArgumentMemoryType::Symbol);
    auto kernel = std::make_shared<CudaKernel>(m_ComputeIdGenerator, m_Configuration, data.GetName(), data.GetSource(),
        data.GetTemplatedName(), symbolArguments);

    if (m_KernelCache.GetMaxSize() > 0)
    {
        m_KernelCache.Put(id, kernel);
    }

    return kernel;
}

std::vector<CUdeviceptr*> CudaEngine::GetKernelArguments(const std::vector<KernelArgument*>& arguments)
{
    std::vector<CUdeviceptr*> result;

    for (auto* argument : arguments)
    {
        if (argument->GetMemoryType() == ArgumentMemoryType::Local || argument->GetMemoryType() == ArgumentMemoryType::Symbol)
        {
            continue;
        }

        CUdeviceptr* deviceArgument = GetKernelArgument(*argument);
        result.push_back(deviceArgument);
    }

    return result;
}

CUdeviceptr* CudaEngine::GetKernelArgument(KernelArgument& argument)
{
    switch (argument.GetMemoryType())
    {
    case ArgumentMemoryType::Scalar:
        return static_cast<CUdeviceptr*>(argument.GetData());
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
    case ArgumentMemoryType::Symbol:
        KttError("Local memory and symbol arguments cannot be retrieved as kernel arguments");
        return nullptr;
    default:
        KttError("Unhandled argument memory type value");
        return nullptr;
    }
}

size_t CudaEngine::GetSharedMemorySize(const std::vector<KernelArgument*>& arguments) const
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

std::unique_ptr<CudaBuffer> CudaEngine::CreateBuffer(KernelArgument& argument)
{
    std::unique_ptr<CudaBuffer> buffer;

    switch (argument.GetMemoryLocation())
    {
    case ArgumentMemoryLocation::Undefined:
        KttError("Buffer cannot be created for arguments with undefined memory location");
        break;
    case ArgumentMemoryLocation::Device:
        buffer = std::make_unique<CudaDeviceBuffer>(argument, m_TransferIdGenerator);
        break;
    case ArgumentMemoryLocation::Host:
    case ArgumentMemoryLocation::HostZeroCopy:
        buffer = std::make_unique<CudaHostBuffer>(argument, m_TransferIdGenerator);
        break;
    case ArgumentMemoryLocation::Unified:
        buffer = std::make_unique<CudaUnifiedBuffer>(argument, m_TransferIdGenerator);
        break;
    default:
        KttError("Unhandled argument memory location value");
        break;
    }

    return buffer;
}

std::unique_ptr<CudaBuffer> CudaEngine::CreateUserBuffer(KernelArgument& argument, ComputeBuffer buffer)
{
    std::unique_ptr<CudaBuffer> userBuffer;

    switch (argument.GetMemoryLocation())
    {
    case ArgumentMemoryLocation::Undefined:
        KttError("Buffer cannot be created for arguments with undefined memory location");
        break;
    case ArgumentMemoryLocation::Device:
        userBuffer = std::make_unique<CudaDeviceBuffer>(argument, m_TransferIdGenerator, buffer);
        break;
    case ArgumentMemoryLocation::Host:
    case ArgumentMemoryLocation::HostZeroCopy:
        userBuffer = std::make_unique<CudaHostBuffer>(argument, m_TransferIdGenerator, buffer);
        break;
    case ArgumentMemoryLocation::Unified:
        userBuffer = std::make_unique<CudaUnifiedBuffer>(argument, m_TransferIdGenerator, buffer);
        break;
    default:
        KttError("Unhandled argument memory location value");
        break;
    }

    return userBuffer;
}

void CudaEngine::InitializeCompilerOptions()
{
    Logger::LogDebug("Initializing default compiler options");

    int computeCapabilityMajor = 0;
    int computeCapabilityMinor = 0;
    CheckError(cuDeviceGetAttribute(&computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, m_Context->GetDevice()),
        "cuDeviceGetAttribute");
    CheckError(cuDeviceGetAttribute(&computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, m_Context->GetDevice()),
        "cuDeviceGetAttribute");

    m_Configuration.SetCompilerOptions("--gpu-architecture=compute_" + std::to_string(computeCapabilityMajor)
        + std::to_string(computeCapabilityMinor));
}

#if defined(KTT_PROFILING_CUPTI)

void CudaEngine::InitializeCupti()
{
    m_Profiler = std::make_unique<CuptiProfiler>();
    m_MetricInterface = std::make_unique<CuptiMetricInterface>(m_DeviceIndex);
}

void CudaEngine::InitializeProfiling(const KernelComputeId& id)
{
    KttAssert(!IsProfilingSessionActive(id), "Attempting to initialize profiling for compute id with active profiling session");
    const auto configuration = m_MetricInterface->CreateMetricConfiguration();
    m_CuptiInstances[id] = std::make_unique<CuptiInstance>(*m_Context, configuration);
}

void CudaEngine::FillProfilingData(const KernelComputeId& id, ComputationResult& result)
{
    KttAssert(IsProfilingSessionActive(id), "Attempting to retrieve profiling data for kernel without active profiling session");
    
    auto& instance = *m_CuptiInstances[id];
    auto profilingData = m_MetricInterface->GenerateProfilingData(instance.GetConfiguration());

    if (profilingData->IsValid())
    {
        KttAssert(instance.HasValidKernelDuration(), "Kernel duration must be known before filling in profiling data");
        uint64_t profiledKernelOverhead = 0;

        if (result.GetDuration() > instance.GetKernelDuration())
        {
            profiledKernelOverhead = result.GetDuration() - instance.GetKernelDuration();
        }

        result.SetDurationData(result.GetDuration() - profiledKernelOverhead, result.GetOverhead() + profiledKernelOverhead);
        m_CuptiInstances.erase(id);
    }

    result.SetProfilingData(std::move(profilingData));
}

#endif // KTT_PROFILING_CUPTI

#if defined(KTT_PROFILING_CUPTI_LEGACY)

void CudaEngine::InitializeProfiling(const KernelComputeId& id)
{
    KttAssert(!IsProfilingSessionActive(id), "Attempting to initialize profiling for compute id with active profiling session");
    m_CuptiInstances[id] = std::make_unique<CuptiInstance>(*m_Context);
}

void CudaEngine::FillProfilingData(const KernelComputeId& id, ComputationResult& result)
{
    KttAssert(IsProfilingSessionActive(id), "Attempting to retrieve profiling data for kernel without active profiling session");
    
    auto& instance = *m_CuptiInstances[id];
    auto profilingData = instance.GenerateProfilingData();

    if (profilingData->IsValid())
    {
        uint64_t profiledKernelOverhead = 0;

        if (result.GetDuration() > instance.GetKernelDuration())
        {
            profiledKernelOverhead = result.GetDuration() - instance.GetKernelDuration();
        }

        result.SetDurationData(result.GetDuration() - profiledKernelOverhead, result.GetOverhead() + profiledKernelOverhead);
        m_CuptiInstances.erase(id);
    }

    result.SetProfilingData(std::move(profilingData));
}

#endif // KTT_PROFILING_CUPTI_LEGACY

} // namespace ktt

#endif // KTT_API_CUDA
