#include <Api/KttException.h>
#include <ComputeEngine/Cuda/CudaEngine.h>
#include <ComputeEngine/OpenCl/OpenClEngine.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/Logger/Logger.h>
#include <TunerCore.h>

namespace ktt
{

TunerCore::TunerCore(const PlatformIndex platform, const DeviceIndex device, const ComputeApi api, const uint32_t queueCount) :
    m_ArgumentManager(std::make_unique<KernelArgumentManager>()),
    m_KernelManager(std::make_unique<KernelManager>(*m_ArgumentManager))
{
    InitializeComputeEngine(platform, device, api, queueCount);
    InitializeRunners();
}

TunerCore::TunerCore(const ComputeApi api, const ComputeApiInitializer& initializer) :
    m_ArgumentManager(std::make_unique<KernelArgumentManager>()),
    m_KernelManager(std::make_unique<KernelManager>(*m_ArgumentManager))
{
    InitializeComputeEngine(api, initializer);
    InitializeRunners();
}

KernelDefinitionId TunerCore::AddKernelDefinition(const std::string& name, const std::string& source, const DimensionVector& globalSize,
    const DimensionVector& localSize)
{
    return m_KernelManager->AddKernelDefinition(name, source, globalSize, localSize);
}

KernelDefinitionId TunerCore::AddKernelDefinitionFromFile(const std::string& name, const std::string& filePath,
    const DimensionVector& globalSize, const DimensionVector& localSize)
{
    return m_KernelManager->AddKernelDefinitionFromFile(name, filePath, globalSize, localSize);
}

void TunerCore::SetArguments(const KernelDefinitionId id, const std::vector<ArgumentId>& argumentIds)
{
    m_KernelManager->SetArguments(id, argumentIds);
}

KernelId TunerCore::CreateKernel(const KernelDefinitionId definitionId)
{
    return m_KernelManager->CreateKernel({definitionId});
}

KernelId TunerCore::CreateKernel(const std::vector<KernelDefinitionId>& definitionIds, KernelLauncher launcher)
{
    const KernelId id = m_KernelManager->CreateKernel(definitionIds);
    m_KernelManager->SetLauncher(id, launcher);
    return id;
}

void TunerCore::SetLauncher(const KernelId id, KernelLauncher launcher)
{
    m_KernelManager->SetLauncher(id, launcher);
}

void TunerCore::AddParameter(const KernelId id, const std::string& name, const std::vector<uint64_t>& values, const std::string& group)
{
    m_KernelManager->AddParameter(id, name, values, group);
}

void TunerCore::AddParameter(const KernelId id, const std::string& name, const std::vector<double>& values, const std::string& group)
{
    m_KernelManager->AddParameter(id, name, values, group);
}

void TunerCore::AddConstraint(const KernelId id, const std::vector<std::string>& parameters,
    std::function<bool(const std::vector<size_t>&)> function)
{
    m_KernelManager->AddConstraint(id, parameters, function);
}

void TunerCore::SetThreadModifier(const KernelId id, const ModifierType type, const ModifierDimension dimension,
    const std::vector<std::string>& parameters, const std::vector<KernelDefinitionId>& definitionIds,
    std::function<uint64_t(const uint64_t, const std::vector<uint64_t>&)> function)
{
    m_KernelManager->SetThreadModifier(id, type, dimension, parameters, definitionIds, function);
}

void TunerCore::SetProfiledDefinitions(const KernelId id, const std::vector<KernelDefinitionId>& definitionIds)
{
    if (definitionIds.size() > 1 && !m_ComputeEngine->SupportsMultiInstanceProfiling())
    {
        throw KttException("The current profiling API does not support profiling of multiple kernel definitions");
    }

    m_KernelManager->SetProfiledDefinitions(id, definitionIds);
}

ArgumentId TunerCore::AddArgumentWithReferencedData(const size_t elementSize, const ArgumentDataType dataType,
    const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const ArgumentMemoryType memoryType,
    const ArgumentManagementType managementType, void* data, const size_t dataSize)
{
    return m_ArgumentManager->AddArgumentWithReferencedData(elementSize, dataType, memoryLocation, accessType, memoryType,
        managementType, data, dataSize);
}

ArgumentId TunerCore::AddArgumentWithOwnedData(const size_t elementSize, const ArgumentDataType dataType,
    const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const ArgumentMemoryType memoryType,
    const ArgumentManagementType managementType, const void* data, const size_t dataSize)
{
    return m_ArgumentManager->AddArgumentWithOwnedData(elementSize, dataType, memoryLocation, accessType, memoryType,
        managementType, data, dataSize);
}

ArgumentId TunerCore::AddUserArgument(ComputeBuffer buffer, const size_t elementSize, const ArgumentDataType dataType,
    const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const size_t dataSize)
{
    const ArgumentId id = m_ArgumentManager->AddUserArgument(elementSize, dataType, memoryLocation, accessType, dataSize);
    auto& argument = m_ArgumentManager->GetArgument(id);
    m_ComputeEngine->AddCustomBuffer(argument, buffer);
    return id;
}

void TunerCore::SetReadOnlyArgumentCache(const bool flag)
{
    m_KernelRunner->SetReadOnlyArgumentCache(flag);
}

KernelResult TunerCore::RunKernel(const KernelId id, const KernelConfiguration& configuration,
    const std::vector<BufferOutputDescriptor>& output)
{
    const auto& kernel = m_KernelManager->GetKernel(id);
    return m_KernelRunner->RunKernel(kernel, configuration, KernelRunMode::Running, output);
}

void TunerCore::SetProfiling(const bool flag)
{
    m_KernelRunner->SetProfiling(flag);
}

void TunerCore::SetValidationMethod(const ValidationMethod method, const double toleranceThreshold)
{
    m_KernelRunner->SetValidationMethod(method, toleranceThreshold);
}

void TunerCore::SetValidationMode(const ValidationMode mode)
{
    m_KernelRunner->SetValidationMode(mode);
}

void TunerCore::SetValidationRange(const ArgumentId id, const size_t range)
{
    m_KernelRunner->SetValidationRange(id, range);
}

void TunerCore::SetValueComparator(const ArgumentId id, ValueComparator comparator)
{
    m_KernelRunner->SetValueComparator(id, comparator);
}

void TunerCore::SetReferenceComputation(const ArgumentId id, ReferenceComputation computation)
{
    m_KernelRunner->SetReferenceComputation(id, computation);
}

void TunerCore::SetReferenceKernel(const ArgumentId id, const KernelId referenceId, const KernelConfiguration& configuration)
{
    const auto& kernel = m_KernelManager->GetKernel(referenceId);
    m_KernelRunner->SetReferenceKernel(id, kernel, configuration);
}

std::vector<KernelResult> TunerCore::TuneKernel(const KernelId id, std::unique_ptr<StopCondition> stopCondition)
{
    const auto& kernel = m_KernelManager->GetKernel(id);
    return m_TuningRunner->Tune(kernel, std::move(stopCondition));
}

KernelResult TunerCore::TuneKernelIteration(const KernelId id, const std::vector<BufferOutputDescriptor>& output, const bool recomputeReference)
{
    const auto& kernel = m_KernelManager->GetKernel(id);
    return m_TuningRunner->TuneIteration(kernel, KernelRunMode::OnlineTuning, output, recomputeReference);
}

void TunerCore::DryTuneKernel(const KernelId id, const std::vector<KernelResult>& results, const uint64_t iterations)
{
    const auto& kernel = m_KernelManager->GetKernel(id);
    m_TuningRunner->DryTune(kernel, results, iterations);
}

void TunerCore::SetSearcher(const KernelId id, std::unique_ptr<Searcher> searcher)
{
    m_TuningRunner->SetSearcher(id, std::move(searcher));
}

void TunerCore::ClearData(const KernelId id)
{
    m_TuningRunner->ClearData(id);
}

const KernelConfiguration& TunerCore::GetBestConfiguration(const KernelId id) const
{
    return m_TuningRunner->GetBestConfiguration(id);
}

KernelConfiguration TunerCore::CreateConfiguration(const KernelId id, const ParameterInput& parameters) const
{
    const auto& kernel = m_KernelManager->GetKernel(id);
    return kernel.CreateConfiguration(parameters);
}

std::string TunerCore::GetKernelSource(const KernelId id, const KernelConfiguration& configuration) const
{
    const auto& kernel = m_KernelManager->GetKernel(id);

    if (kernel.IsComposite())
    {
        throw KttException("Kernel with id " + std::to_string(id) + " has multiple definitions, its source cannot be directly retrieved");
    }

    const auto& definition = kernel.GetPrimaryDefinition();
    return configuration.GeneratePrefix() + definition.GetSource();
}

std::string TunerCore::GetKernelDefinitionSource(const KernelDefinitionId id, const KernelConfiguration& configuration) const
{
    const auto& definition = m_KernelManager->GetDefinition(id);
    return configuration.GeneratePrefix() + definition.GetSource();
}

void TunerCore::SetTimeUnit(const TimeUnit unit)
{
    m_KernelRunner->SetTimeUnit(unit);
}

void TunerCore::SetProfilingCounters(const std::vector<std::string>& counters)
{
    m_ComputeEngine->SetProfilingCounters(counters);
}

void TunerCore::SetCompilerOptions(const std::string& options)
{
    m_ComputeEngine->SetCompilerOptions(options);
}

void TunerCore::SetGlobalSizeType(const GlobalSizeType type)
{
    m_ComputeEngine->SetGlobalSizeType(type);
}

void TunerCore::SetAutomaticGlobalSizeCorrection(const bool flag)
{
    m_ComputeEngine->SetAutomaticGlobalSizeCorrection(flag);
}

void TunerCore::SetKernelCacheCapacity(const uint64_t capacity)
{
    m_ComputeEngine->SetKernelCacheCapacity(capacity);
}

std::vector<PlatformInfo> TunerCore::GetPlatformInfo() const
{
    return m_ComputeEngine->GetPlatformInfo();
}

std::vector<DeviceInfo> TunerCore::GetDeviceInfo(const PlatformIndex platform) const
{
    return m_ComputeEngine->GetDeviceInfo(platform);
}

DeviceInfo TunerCore::GetCurrentDeviceInfo() const
{
    return m_ComputeEngine->GetCurrentDeviceInfo();
}

void TunerCore::SetLoggingLevel(const LoggingLevel level)
{
    Logger::GetLogger().SetLoggingLevel(level);
}

void TunerCore::SetLoggingTarget(std::ostream& target)
{
    Logger::GetLogger().SetLoggingTarget(target);
}

void TunerCore::SetLoggingTarget(const std::string& file)
{
    Logger::GetLogger().SetLoggingTarget(file);
}

void TunerCore::Log(const LoggingLevel level, const std::string& message)
{
    Logger::GetLogger().Log(level, message);
}

void TunerCore::InitializeComputeEngine(const PlatformIndex platform, const DeviceIndex device, const ComputeApi api, const uint32_t queueCount)
{
    if (queueCount == 0)
    {
        throw KttException("Number of compute queues must be greater than zero");
    }

    switch (api)
    {
    case ComputeApi::OpenCL:
        #ifdef KTT_API_OPENCL
        m_ComputeEngine = std::make_unique<OpenClEngine>(platform, device, queueCount);
        #else
        throw KttException("Support for OpenCL API is not included in this version of KTT framework");
        #endif // KTT_API_OPENCL
        break;
    case ComputeApi::CUDA:
        #ifdef KTT_API_CUDA
        m_ComputeEngine = std::make_unique<CudaEngine>(device, queueCount);
        #else
        throw KttException("Support for CUDA API is not included in this version of KTT framework");
        #endif // KTT_API_CUDA
        break;
    case ComputeApi::Vulkan:
        #ifdef KTT_API_VULKAN
        throw KttException("Support for Vulkan API is not yet available");
        #else
        throw KttException("Support for Vulkan API is not included in this version of KTT framework");
        #endif // KTT_API_VULKAN
        break;
    default:
        KttError("Unhandled compute API value");
    }
}

void TunerCore::InitializeComputeEngine(const ComputeApi api, const ComputeApiInitializer& initializer)
{
    switch (api)
    {
    case ComputeApi::OpenCL:
        #ifdef KTT_API_OPENCL
        m_ComputeEngine = std::make_unique<OpenClEngine>(initializer);
        #else
        throw KttException("Support for OpenCL API is not included in this version of KTT framework");
        #endif // KTT_API_OPENCL
        break;
    case ComputeApi::CUDA:
        #ifdef KTT_API_CUDA
        m_ComputeEngine = std::make_unique<CudaEngine>(initializer);
        #else
        throw KttException("Support for CUDA API is not included in this version of KTT framework");
        #endif // KTT_API_CUDA
        break;
    case ComputeApi::Vulkan:
        #ifdef KTT_API_VULKAN
        throw KttException("Support for user initializers is not yet available for Vulkan API");
        #else
        throw KttException("Support for Vulkan API is not included in this version of KTT framework");
        #endif // KTT_API_VULKAN
        break;
    default:
        KttError("Unhandled compute API value");
    }
}

void TunerCore::InitializeRunners()
{
    DeviceInfo info = m_ComputeEngine->GetCurrentDeviceInfo();
    Logger::LogInfo("Initializing tuner for device " + info.GetName());

    m_KernelRunner = std::make_unique<KernelRunner>(*m_ComputeEngine, *m_ArgumentManager);
    m_TuningRunner = std::make_unique<TuningRunner>(*m_KernelRunner, info);
}

} // namespace ktt
