#ifdef KTT_API_CPP

#include <Api/KttException.h>
#include <ComputeEngine/Cpp/CppEngine.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/Logger/Logger.h>
#include <Utility/Timer/Timer.h>
#include <Utility/StlHelpers.h>
#include <Utility/StringUtility.h>
#include <cstring>
#include <future>
#include <fstream>

#if defined(_MSC_VER)
#include <windows.h>
#include <intrin.h>
#else
#include <unistd.h>
#endif

namespace ktt
{

CppEngine::CppEngine(const PlatformIndex platformIndex, const DeviceIndex deviceIndex, const uint32_t queueCount) :
    m_Configuration(GlobalSizeType::OpenCL), // TODO: decide appropriate global size type for C++
    m_PlatformIndex(platformIndex),
    m_DeviceIndex(deviceIndex),
    m_DeviceInfo(0, ""),
    m_KernelCache(10)
{
    // For CPU backend, we ignore platform and device indices.
    if (queueCount == 0)
    {
        throw KttException("Number of compute queues must be greater than zero");
    }

    for (uint32_t i = 0; i < queueCount; ++i)
    {
        const QueueId id = m_QueueIdGenerator.GenerateId();
        // Each queue is just a placeholder; we don't create threads yet.
        m_Queues[id] = QueueData{};
    }

    // Set device info
    m_DeviceInfo = DeviceInfo(0, "CPU");
}

CppEngine::CppEngine(const ComputeApiInitializer& initializer, std::vector<QueueId>& assignedQueueIds) :
    m_Configuration(GlobalSizeType::OpenCL),
    m_DeviceInfo(0, ""),
    m_KernelCache(10)
{
    // Silence unused parameter warnings - custom initializer not supported for C++ backend
    (void)initializer;
    (void)assignedQueueIds;
    
    // Custom initializer not supported for C++ backend.
    throw KttException("Support for user initializers is not available for C++ API");
}

ComputeActionId CppEngine::RunKernelAsync(const KernelComputeData& data, const QueueId queueId, const bool powerMeasurementAllowed)
{
    // Silence unused parameter warnings - queues are not used in C++ backend
    (void)queueId;
    (void)powerMeasurementAllowed;
    Timer overheadTimer;
    overheadTimer.Start();

    // Load kernel (cached)
    auto kernel = LoadKernel(data);
    // Set kernel arguments based on current buffers
    SetKernelArguments(*kernel, data.GetArguments());

    overheadTimer.Stop();
    const Nanoseconds overhead = overheadTimer.GetElapsedTime();

    const ComputeActionId id = m_ComputeIdGenerator.GenerateId();
    const KernelComputeId computeId = data.GetUniqueIdentifier();
    m_ComputeActionToKernel[id] = computeId;

    // Capture kernel name for the result
    const std::string kernelName = data.GetName();

    // Launch kernel asynchronously
    m_ComputeActions[id] = std::async(std::launch::async, [kernel, overhead, kernelName]() {
        Timer timer;
        timer.Start();

        // Execute kernel
        kernel->function(kernel->argumentPointers, kernel->argumentSizes);

        timer.Stop();
        ComputationResult result(kernelName);
        // overhead includes kernel loading and argument setup
        // compilation overhead is tracked separately in LoadKernel when compilation happens
        result.SetDurationData(timer.GetElapsedTime(), overhead, kernel->compilationOverhead);
        return result;
    });

    return id;
}

ComputationResult CppEngine::WaitForComputeAction(const ComputeActionId id)
{
    auto it = m_ComputeActions.find(id);
    if (it == m_ComputeActions.end())
    {
        throw KttException("Compute action with id " + std::to_string(id) + " not found");
    }
    ComputationResult result = it->second.get();
    m_ComputeActions.erase(it);
    m_ComputeActionToKernel.erase(id);
    return result;
}

void CppEngine::ClearData(const KernelComputeId& id)
{
    // Remove compute actions associated with this kernel compute id
    std::vector<ComputeActionId> toRemove;
    for (const auto& pair : m_ComputeActionToKernel)
    {
        if (pair.second == id)
        {
            toRemove.push_back(pair.first);
        }
    }
    for (const ComputeActionId actionId : toRemove)
    {
        m_ComputeActions.erase(actionId);
        m_ComputeActionToKernel.erase(actionId);
    }
}

void CppEngine::ClearKernelData(const std::string& kernelName)
{
    // Remove compute actions associated with kernels matching the name
    std::vector<ComputeActionId> actionsToRemove;
    for (const auto& pair : m_ComputeActionToKernel)
    {
        if (StartsWith(pair.second, kernelName))
        {
            actionsToRemove.push_back(pair.first);
        }
    }
    for (const ComputeActionId actionId : actionsToRemove)
    {
        m_ComputeActions.erase(actionId);
        m_ComputeActionToKernel.erase(actionId);
    }

    // Remove kernels with matching name from cache
    // The cache key is KernelComputeId (string) which includes the kernel name
    // Since LruCache doesn't support EraseIf, we need to iterate and remove manually
    std::vector<KernelComputeId> kernelsToRemove;
    for (auto it = m_KernelCache.Begin(); it != m_KernelCache.End(); ++it)
    {
        if (StartsWith(it->first, kernelName))
        {
            kernelsToRemove.push_back(it->first);
        }
    }
    for (const auto& computeId : kernelsToRemove)
    {
        m_KernelCache.Remove(computeId);
    }
}

// Profiling methods - not supported for CPU backend
ComputationResult CppEngine::RunKernelWithProfiling(const KernelComputeData& data, const QueueId queueId)
{
    (void)data;
    (void)queueId;
    throw KttException("Profiling is not supported for C++ backend");
}

void CppEngine::SetProfilingCounters(const std::vector<std::string>& counters)
{
    // Silence unused parameter warning - profiling not supported for C++ backend
    (void)counters;
    throw KttException("Profiling is not supported for C++ backend");
}

bool CppEngine::IsProfilingSessionActive(const KernelComputeId& id)
{
    // Silence unused parameter warning - profiling not supported for C++ backend
    (void)id;
    return false;
}

uint64_t CppEngine::GetRemainingProfilingRuns(const KernelComputeId& id)
{
    // Silence unused parameter warning - profiling not supported for C++ backend
    (void)id;
    return 0;
}

bool CppEngine::HasAccurateRemainingProfilingRuns() const
{
    return false;
}

bool CppEngine::SupportsMultiInstanceProfiling() const
{
    return false;
}

bool CppEngine::IsProfilingActive() const
{
    return false;
}

void CppEngine::SetProfiling(const bool profiling)
{
    // Silence unused parameter warning - profiling not supported for C++ backend
    (void)profiling;
    if (profiling) {
        throw KttException("Profiling is not supported for C++ backend");
    }
}

// Buffer methods
TransferActionId CppEngine::UploadArgument(KernelArgument& kernelArgument, const QueueId queueId)
{
    (void)queueId;
    Timer timer;
    timer.Start();

    const ArgumentId& id = kernelArgument.GetId();
    
    // Check if buffer already exists (e.g., from AddCustomBuffer for zero-copy)
    if (m_Buffers.find(id) == m_Buffers.end())
    {
        // Create buffer in CPU memory only if it doesn't exist
        auto buffer = CreateBuffer(kernelArgument);
        m_Buffers[id] = *buffer;
    }
    // If buffer already exists (from AddCustomBuffer), use it as-is (zero-copy)

    timer.Stop();
    const Nanoseconds overhead = timer.GetElapsedTime();

    // Simulate transfer action using std::promise to avoid thread creation overhead
    const TransferActionId actionId = m_TransferIdGenerator.GenerateId();
    std::promise<TransferResult> promise;
    TransferResult result;
    result.SetDuration(0);
    result.SetOverhead(overhead);
    promise.set_value(result);
    m_TransferActions[actionId] = promise.get_future();
    return actionId;
}

TransferActionId CppEngine::UpdateArgument(const ArgumentId& id, const QueueId queueId, const void* data,
    const size_t dataSize)
{
    (void)queueId;
    Timer timer;
    timer.Start();

    auto it = m_Buffers.find(id);
    if (it == m_Buffers.end())
    {
        throw KttException("Buffer with id " + id + " not found");
    }
    CppBuffer& buffer = it->second;
    if (dataSize > buffer.size)
    {
        throw KttException("Update data size exceeds buffer size");
    }
    memcpy(buffer.data, data, dataSize);

    timer.Stop();
    const Nanoseconds overhead = timer.GetElapsedTime();

    const TransferActionId actionId = m_TransferIdGenerator.GenerateId();
    std::promise<TransferResult> promise;
    TransferResult result;
    result.SetDuration(0);
    result.SetOverhead(overhead);
    promise.set_value(result);
    m_TransferActions[actionId] = promise.get_future();
    return actionId;
}

TransferActionId CppEngine::DownloadArgument(const ArgumentId& id, const QueueId queueId, void* destination,
    const size_t dataSize)
{
    (void)queueId;
    Timer timer;
    timer.Start();

    auto it = m_Buffers.find(id);
    if (it == m_Buffers.end())
    {
        throw KttException("Buffer with id " + id + " not found");
    }
    CppBuffer& buffer = it->second;
    if (dataSize > buffer.size)
    {
        throw KttException("Download data size exceeds buffer size");
    }
    memcpy(destination, buffer.data, dataSize);

    timer.Stop();
    const Nanoseconds overhead = timer.GetElapsedTime();

    const TransferActionId actionId = m_TransferIdGenerator.GenerateId();
    std::promise<TransferResult> promise;
    TransferResult result;
    result.SetDuration(0);
    result.SetOverhead(overhead);
    promise.set_value(result);
    m_TransferActions[actionId] = promise.get_future();
    return actionId;
}

TransferActionId CppEngine::CopyArgument(const ArgumentId& destination, const QueueId queueId, const ArgumentId& source,
    const size_t dataSize)
{
    (void)queueId;
    Timer timer;
    timer.Start();

    auto srcIt = m_Buffers.find(source);
    auto dstIt = m_Buffers.find(destination);
    if (srcIt == m_Buffers.end() || dstIt == m_Buffers.end())
    {
        throw KttException("Source or destination buffer not found");
    }
    CppBuffer& src = srcIt->second;
    CppBuffer& dst = dstIt->second;
    size_t copySize = std::min({src.size, dst.size, dataSize});
    memcpy(dst.data, src.data, copySize);

    timer.Stop();
    const Nanoseconds overhead = timer.GetElapsedTime();

    const TransferActionId actionId = m_TransferIdGenerator.GenerateId();
    std::promise<TransferResult> promise;
    TransferResult result;
    result.SetDuration(0);
    result.SetOverhead(overhead);
    promise.set_value(result);
    m_TransferActions[actionId] = promise.get_future();
    return actionId;
}

TransferResult CppEngine::WaitForTransferAction(const TransferActionId id)
{
    auto it = m_TransferActions.find(id);
    if (it == m_TransferActions.end())
    {
        throw KttException("Transfer action with id " + std::to_string(id) + " not found");
    }
    TransferResult result = it->second.get();
    m_TransferActions.erase(it);
    return result;
}

void CppEngine::ResizeArgument(const ArgumentId& id, const size_t newSize, const bool preserveData)
{
    auto it = m_Buffers.find(id);
    if (it == m_Buffers.end())
    {
        throw KttException("Buffer with id " + id + " not found");
    }
    CppBuffer& buffer = it->second;
    void* newData = malloc(newSize);
    if (preserveData)
    {
        size_t copySize = std::min(buffer.size, newSize);
        memcpy(newData, buffer.data, copySize);
    }
    // Free old data and assign new data
    free(buffer.data);
    buffer.data = newData;
    buffer.size = newSize;
}

void CppEngine::GetUnifiedMemoryBufferHandle(const ArgumentId& id, UnifiedBufferMemory& handle)
{
    auto it = m_Buffers.find(id);
    if (it == m_Buffers.end())
    {
        throw KttException("Buffer with id " + id + " not found");
    }
    handle = it->second.data;
}

void CppEngine::AddCustomBuffer(KernelArgument& kernelArgument, ComputeBuffer buffer)
{
    auto cppBuffer = CreateUserBuffer(kernelArgument, buffer);
    const ArgumentId& id = kernelArgument.GetId();
    // Use move semantics to avoid copy constructor which would allocate new memory
    m_Buffers[id] = std::move(*cppBuffer);
}

void CppEngine::ClearBuffer(const ArgumentId& id)
{
    auto it = m_Buffers.find(id);
    if (it != m_Buffers.end())
    {
        m_Buffers.erase(it);
    }
}

void CppEngine::ClearBuffers()
{
    m_Buffers.clear();
}

bool CppEngine::HasBuffer(const ArgumentId& id)
{
    return m_Buffers.find(id) != m_Buffers.end();
}

// Queue methods
QueueId CppEngine::AddComputeQueue(ComputeQueue queue)
{
    // Silence unused parameter warning - custom queues not supported for C++ backend
    (void)queue;
    
    // Not yet supported
    throw KttException("Adding custom compute queues is not supported for C++ backend");
}

void CppEngine::RemoveComputeQueue(const QueueId id)
{
    auto it = m_Queues.find(id);
    if (it == m_Queues.end())
    {
        throw KttException("Queue with id " + std::to_string(id) + " not found");
    }
    // TODO: ensure no pending actions
    m_Queues.erase(it);
}

QueueId CppEngine::GetDefaultQueue() const
{
    if (m_Queues.empty())
    {
        throw KttException("No compute queues available");
    }
    return m_Queues.begin()->first;
}

std::vector<QueueId> CppEngine::GetAllQueues() const
{
    std::vector<QueueId> result;
    for (const auto& pair : m_Queues)
    {
        result.push_back(pair.first);
    }
    return result;
}

void CppEngine::SynchronizeQueue(const QueueId queueId)
{
    (void)queueId;
    // For CPU backend, synchronization is immediate.
    // If we had pending actions per queue, we would wait for them.
    Logger::LogWarning("Synchronizing queue is implicit on CPU.");
}

void CppEngine::SynchronizeQueues()
{
    // No-op
    Logger::LogWarning("Synchronizing queue is implicit on CPU.");
}

void CppEngine::SynchronizeDevice()
{
    // No-op
    Logger::LogWarning("Synchronizing device is implicit on CPU.");
}

// Information retrieval methods
std::vector<PlatformInfo> CppEngine::GetPlatformInfo() const
{
    // Return a single CPU platform
    PlatformInfo info(0, "CPU Platform");
    return {info};
}

std::vector<DeviceInfo> CppEngine::GetDeviceInfo(const PlatformIndex platformIndex) const
{
    if (platformIndex != 0)
    {
        return {};
    }

    // Parse /proc/cpuinfo for CPU model and vendor
    std::string cpuName = "CPU";
    std::string vendor = "Generic";

#if defined(_MSC_VER)
    // Windows: Use CPUID to get processor info
    int cpuInfo[4] = {-1};
    char vendorString[13] = {0};
    
    // Get vendor string
    __cpuid(cpuInfo, 0);
    memcpy(vendorString, &cpuInfo[1], 4);
    memcpy(vendorString + 4, &cpuInfo[3], 4);
    memcpy(vendorString + 8, &cpuInfo[2], 4);
    vendorString[12] = '\0';
    vendor = vendorString;
    
    // Get brand string (CPU name) - requires CPUID with eax = 0x80000002, 0x80000003, 0x80000004
    char brandString[49] = {0};
    __cpuid(cpuInfo, 0x80000000);
    unsigned int maxExtendedId = cpuInfo[0];
    if (maxExtendedId >= 0x80000004)
    {
        __cpuid(cpuInfo, 0x80000002);
        memcpy(brandString, cpuInfo, 16);
        __cpuid(cpuInfo, 0x80000003);
        memcpy(brandString + 16, cpuInfo, 16);
        __cpuid(cpuInfo, 0x80000004);
        memcpy(brandString + 32, cpuInfo, 16);
        brandString[48] = '\0';
        cpuName = brandString;
        // Trim leading whitespace
        size_t start = cpuName.find_first_not_of(" ");
        if (start != std::string::npos)
        {
            cpuName = cpuName.substr(start);
        }
    }
#else
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line))
    {
        if (line.find("model name") == 0)
        {
            size_t pos = line.find(':');
            if (pos != std::string::npos)
            {
                cpuName = line.substr(pos + 2);
            }
        }
        if (line.find("vendor_id") == 0)
        {
            size_t pos = line.find(':');
            if (pos != std::string::npos)
            {
                vendor = line.substr(pos + 2);
            }
        }
    }
#endif

    // Get memory size
    uint64_t memorySize = 0;
#if defined(_MSC_VER)
    // Windows: Use GlobalMemoryStatusEx
    MEMORYSTATUSEX memStatus;
    memStatus.dwLength = sizeof(memStatus);
    if (GlobalMemoryStatusEx(&memStatus))
    {
        memorySize = memStatus.ullTotalPhys;
    }
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long pageSize = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && pageSize > 0)
    {
        memorySize = static_cast<uint64_t>(pages) * static_cast<uint64_t>(pageSize);
    }
#endif

    DeviceInfo info(0, cpuName);
    info.SetVendor(vendor);
    info.SetExtensions("C++");
    info.SetGlobalMemorySize(memorySize);
    info.SetLocalMemorySize(0);
    info.SetMaxWorkGroupSize(1024); // arbitrary
    info.SetMaxConstantBufferSize(0);
    info.SetMaxComputeUnits(static_cast<uint32_t>(std::thread::hardware_concurrency()));
    return {info};
}

PlatformInfo CppEngine::GetCurrentPlatformInfo() const
{
    return PlatformInfo(0, "CPU Platform");
}

DeviceInfo CppEngine::GetCurrentDeviceInfo() const
{
    return m_DeviceInfo;
}

ComputeApi CppEngine::GetComputeApi() const
{
    return ComputeApi::Cpp;
}

GlobalSizeType CppEngine::GetGlobalSizeType() const
{
    return m_Configuration.GetGlobalSizeType();
}

// Utility methods
void CppEngine::SetCompilerOptions(const std::string& options, const bool overrideDefault)
{
    std::string finalOptions = options;

    if (!overrideDefault)
    {
        if (!finalOptions.empty())
        {
            finalOptions += " ";
        }

        finalOptions += m_Configuration.GetDefaultCompilerOptions();
    }

    m_Configuration.SetCompilerOptions(finalOptions);
    ClearKernelCache();
}

void CppEngine::SetCompiler(const std::string& compiler)
{
    m_Compiler.SetCompiler(compiler);
    ClearKernelCache();
}

void CppEngine::SetGlobalSizeType(const GlobalSizeType type)
{
    m_Configuration.SetGlobalSizeType(type);
}

void CppEngine::SetAutomaticGlobalSizeCorrection(const bool flag)
{
    m_Configuration.SetGlobalSizeCorrection(flag);
}

void CppEngine::SetKernelCacheCapacity(const uint64_t capacity)
{
    m_KernelCache.SetMaxSize(capacity);
}

void CppEngine::ClearKernelCache()
{
    m_KernelCache.Clear();
}

void CppEngine::EnsureThreadContext()
{
    // No-op
}

// Private methods
std::shared_ptr<CppEngine::CppKernel> CppEngine::LoadKernel(const KernelComputeData& data)
{
    const KernelComputeId& computeId = data.GetUniqueIdentifier();
    
    // Check cache first
    if (m_KernelCache.GetMaxSize() > 0 && m_KernelCache.Exists(computeId))
    {
        return m_KernelCache.Get(computeId)->second;
    }

    // Compile kernel
    const std::string kernelName = data.GetName();
    const std::string kernelSource = data.GetSource();
    const std::string compilerOptions = m_Configuration.GetCompilerOptions();

    Timer compilationTimer;
    compilationTimer.Start();

    CppCompiler::KernelFunction function;
    try
    {
        function = m_Compiler.CompileKernel(kernelName, kernelSource, compilerOptions);
    }
    catch (const KttException& e)
    {
        Logger::LogError("Failed to compile kernel " + kernelName + ": " + e.what());
        throw;
    }

    compilationTimer.Stop();

    auto kernel = std::make_shared<CppKernel>();
    kernel->function = std::move(function);
    kernel->compilationOverhead = compilationTimer.GetElapsedTime();
    // argument pointers and sizes will be filled later by SetKernelArguments

    // Cache the compiled kernel
    if (m_KernelCache.GetMaxSize() > 0)
    {
        m_KernelCache.Put(computeId, kernel);
    }

    return kernel;
}

void CppEngine::SetKernelArguments(CppKernel& kernel, const std::vector<KernelArgument*> arguments)
{
    kernel.argumentPointers.clear();
    kernel.argumentSizes.clear();
    kernel.scalarValues.clear();
    
    // First pass: count non-scalar arguments for pointers array
    size_t nonScalarCount = 0;
    for (const auto* arg : arguments)
    {
        if (arg != nullptr && arg->GetMemoryType() != ArgumentMemoryType::Scalar)
        {
            ++nonScalarCount;
        }
    }
    
    kernel.argumentPointers.reserve(nonScalarCount);
    kernel.argumentSizes.reserve(arguments.size());
    kernel.scalarValues.reserve(arguments.size());

    for (const auto* arg : arguments)
    {
        if (arg == nullptr)
        {
            throw KttException("Null kernel argument encountered");
        }
        
        const ArgumentId id = arg->GetId();
        
        // Handle scalar arguments - store their value directly
        if (arg->GetMemoryType() == ArgumentMemoryType::Scalar)
        {
            // For scalar arguments, we store the value in scalarValues
            // and put a pointer to it in argumentSizes
            size_t scalarValue = 0;
            if (arg->GetDataSize() <= sizeof(size_t))
            {
                std::memcpy(&scalarValue, arg->GetData(), arg->GetDataSize());
            }
            kernel.scalarValues.push_back(scalarValue);
            kernel.argumentSizes.push_back(scalarValue);
        }
        else
        {
            // Handle vector/buffer arguments
            auto it = m_Buffers.find(id);
            if (it == m_Buffers.end())
            {
                throw KttException("Buffer for argument id " + id + " not found");
            }
            kernel.argumentPointers.push_back(it->second.data);
            kernel.argumentSizes.push_back(it->second.size);
        }
    }
}

void CppEngine::SetKernelArgument(CppKernel& kernel, const KernelArgument& argument)
{
    const ArgumentId id = argument.GetId();
    auto it = m_Buffers.find(id);
    if (it == m_Buffers.end())
    {
        throw KttException("Buffer for argument id " + id + " not found");
    }
    // Find if argument already exists in kernel (by id) and replace, else append.
    // For simplicity, we assume arguments are set in order; we'll just append.
    kernel.argumentPointers.push_back(it->second.data);
    kernel.argumentSizes.push_back(it->second.size);
}

std::unique_ptr<CppEngine::CppBuffer> CppEngine::CreateBuffer(KernelArgument& argument)
{
    auto buffer = std::make_unique<CppBuffer>();
    buffer->size = argument.GetDataSize();
    buffer->data = malloc(buffer->size);
    buffer->location = argument.GetMemoryLocation();
    buffer->access = argument.GetAccessType();
    // If argument has initial data, copy it
    if (argument.HasOwnedData())
    {
        memcpy(buffer->data, argument.GetData(), buffer->size);
    }
    return buffer;
}

std::unique_ptr<CppEngine::CppBuffer> CppEngine::CreateUserBuffer(KernelArgument& argument, ComputeBuffer userBuffer)
{
    auto buffer = std::make_unique<CppBuffer>();
    buffer->size = argument.GetDataSize();
    buffer->data = userBuffer;
    buffer->location = argument.GetMemoryLocation();
    buffer->access = argument.GetAccessType();
    buffer->owned = false; // User-provided buffer, don't free it
    return buffer;
}

void CppEngine::ClearQueueActions(const QueueId id)
{
    // Silence unused parameter warning - per-queue action tracking not implemented for C++ backend
    (void)id;
    // TODO: implement if we track per-queue actions
}

} // namespace ktt

#endif // KTT_API_CPP
