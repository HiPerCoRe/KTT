#include <fstream>

#include <Api/KttException.h>
#include <ComputeEngine/Cuda/CudaEngine.h>
#include <ComputeEngine/OpenCl/OpenClEngine.h>
#include <ComputeEngine/Vulkan/VulkanEngine.h>
#include <Output/Deserializer/JsonDeserializer.h>
#include <Output/Deserializer/XmlDeserializer.h>
#include <Output/Serializer/JsonSerializer.h>
#include <Output/Serializer/XmlSerializer.h>
#include <Output/TimeConfiguration/TimeConfiguration.h>
#include <Output/TunerMetadata.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/Logger/Logger.h>
#include <Utility/FileSystem.h>
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

TunerCore::TunerCore(const ComputeApi api, const ComputeApiInitializer& initializer, std::vector<QueueId>& assignedQueueIds) :
    m_ArgumentManager(std::make_unique<KernelArgumentManager>()),
    m_KernelManager(std::make_unique<KernelManager>(*m_ArgumentManager))
{
    InitializeComputeEngine(api, initializer, assignedQueueIds);
    InitializeRunners();
}

KernelDefinitionId TunerCore::AddKernelDefinition(const std::string& name, const std::string& source,
    const DimensionVector& globalSize, const DimensionVector& localSize, const std::vector<std::string>& typeNames)
{
    return m_KernelManager->AddKernelDefinition(name, source, globalSize, localSize, typeNames);
}

KernelDefinitionId TunerCore::AddKernelDefinitionFromFile(const std::string& name, const std::string& filePath,
    const DimensionVector& globalSize, const DimensionVector& localSize, const std::vector<std::string>& typeNames)
{
    return m_KernelManager->AddKernelDefinitionFromFile(name, filePath, globalSize, localSize, typeNames);
}

KernelDefinitionId TunerCore::GetKernelDefinitionId(const std::string& name, const std::vector<std::string>& typeNames) const
{
    return m_KernelManager->GetDefinitionId(name, typeNames);
}

void TunerCore::RemoveKernelDefinition(const KernelDefinitionId id)
{
    const auto& definition = m_KernelManager->GetDefinition(id);
    m_ComputeEngine->ClearKernelData(definition.GetName() + definition.GetTemplatedName());
    m_KernelManager->RemoveKernelDefinition(id);
}

void TunerCore::SetArguments(const KernelDefinitionId id, const std::vector<ArgumentId>& argumentIds)
{
    m_KernelManager->SetArguments(id, argumentIds);
}

KernelId TunerCore::CreateKernel(const std::string& name, const KernelDefinitionId definitionId)
{
    return m_KernelManager->CreateKernel(name, {definitionId});
}

KernelId TunerCore::CreateKernel(const std::string& name, const std::vector<KernelDefinitionId>& definitionIds, KernelLauncher launcher)
{
    const KernelId id = m_KernelManager->CreateKernel(name, definitionIds);
    m_KernelManager->SetLauncher(id, launcher);
    return id;
}

void TunerCore::RemoveKernel(const KernelId id)
{
    m_TuningRunner->ClearConfigurationData(id, true);
    m_KernelRunner->RemoveKernelData(id);
    m_KernelManager->RemoveKernel(id);
}

void TunerCore::SetLauncher(const KernelId id, KernelLauncher launcher)
{
    m_KernelManager->SetLauncher(id, launcher);
}

void TunerCore::AddParameter(const KernelId id, const std::string& name, const std::vector<ParameterValue>& values, const std::string& group)
{
    m_KernelManager->AddParameter(id, name, values, group);
}

void TunerCore::AddScriptParameter(const KernelId id, const std::string& name, const ParameterValueType valueType, const std::string& valueScript,
    const std::string& group)
{
    m_KernelManager->AddScriptParameter(id, name, valueType, valueScript, group);
}

void TunerCore::AddConstraint(const KernelId id, const std::vector<std::string>& parameters, ConstraintFunction function)
{
    m_KernelManager->AddConstraint(id, parameters, function);
}

void TunerCore::AddGenericConstraint(const KernelId id, const std::vector<std::string>& parameters, GenericConstraintFunction function)
{
    m_KernelManager->AddGenericConstraint(id, parameters, function);
}

void TunerCore::AddScriptConstraint(const KernelId id, const std::vector<std::string>& parameters, const std::string& script)
{
    m_KernelManager->AddScriptConstraint(id, parameters, script);
}

void TunerCore::AddThreadModifier(const KernelId id, const std::vector<KernelDefinitionId>& definitionIds, const ModifierType type,
    const ModifierDimension dimension, const std::vector<std::string>& parameters, ModifierFunction function)
{
    m_KernelManager->AddThreadModifier(id, definitionIds, type, dimension, parameters, function);
}

void TunerCore::AddScriptThreadModifier(const KernelId id, const std::vector<KernelDefinitionId>& definitionIds, const ModifierType type,
    const ModifierDimension dimension, const std::string& script)
{
    m_KernelManager->AddScriptThreadModifier(id, definitionIds, type, dimension, script);
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
    const ArgumentManagementType managementType, const void* data, const size_t dataSize, const std::string& symbolName)
{
    if (memoryType == ArgumentMemoryType::Symbol && symbolName.empty() && m_ComputeEngine->GetComputeApi() == ComputeApi::CUDA)
    {
        throw KttException("Symbol arguments in CUDA must have defined symbol name");
    }

    return m_ArgumentManager->AddArgumentWithOwnedData(elementSize, dataType, memoryLocation, accessType, memoryType,
        managementType, data, dataSize, symbolName);
}

ArgumentId TunerCore::AddUserArgument(ComputeBuffer buffer, const size_t elementSize, const ArgumentDataType dataType,
    const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const size_t dataSize)
{
    const ArgumentId id = m_ArgumentManager->AddUserArgument(elementSize, dataType, memoryLocation, accessType, dataSize);
    auto& argument = m_ArgumentManager->GetArgument(id);
    m_ComputeEngine->AddCustomBuffer(argument, buffer);
    return id;
}

void TunerCore::RemoveArgument(const ArgumentId id)
{
    if (m_KernelManager->IsArgumentUsed(id))
    {
        throw KttException("Argument with id " + std::to_string(id) +
            " cannot be removed because it is still referenced by at least one kernel definition");
    }

    m_KernelRunner->RemoveValidationData(id);
    m_ComputeEngine->ClearBuffer(id);
    m_ArgumentManager->RemoveArgument(id);
}

void TunerCore::SaveArgument(const ArgumentId id, const std::string& file) const
{
    m_ArgumentManager->SaveArgument(id, file);
}

void TunerCore::SetReadOnlyArgumentCache(const bool flag)
{
    m_KernelRunner->SetReadOnlyArgumentCache(flag);
}

KernelResult TunerCore::RunKernel(const KernelId id, const KernelConfiguration& configuration, const KernelDimensions& dimensions,
    const std::vector<BufferOutputDescriptor>& output)
{
    const auto& kernel = m_KernelManager->GetKernel(id);
    return m_KernelRunner->RunKernel(kernel, configuration, dimensions, KernelRunMode::Running, output);
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

void TunerCore::SetReferenceKernel(const ArgumentId id, const KernelId referenceId, const KernelConfiguration& configuration,
    const KernelDimensions& dimensions)
{
    const auto& kernel = m_KernelManager->GetKernel(referenceId);
    m_KernelRunner->SetReferenceKernel(id, kernel, configuration, dimensions);
}

std::vector<KernelResult> TunerCore::TuneKernel(const KernelId id, const KernelDimensions& dimensions,
    std::unique_ptr<StopCondition> stopCondition)
{
    const auto& kernel = m_KernelManager->GetKernel(id);
    return m_TuningRunner->Tune(kernel, dimensions, std::move(stopCondition));
}

KernelResult TunerCore::TuneKernelIteration(const KernelId id, const KernelDimensions& dimensions,
    const std::vector<BufferOutputDescriptor>& output, const bool recomputeReference)
{
    const auto& kernel = m_KernelManager->GetKernel(id);
    return m_TuningRunner->TuneIteration(kernel, dimensions, KernelRunMode::OnlineTuning, output, recomputeReference);
}

std::vector<KernelResult> TunerCore::SimulateKernelTuning(const KernelId id, const std::vector<KernelResult>& results,
    const uint64_t iterations)
{
    const auto& kernel = m_KernelManager->GetKernel(id);
    return m_TuningRunner->SimulateTuning(kernel, results, iterations);
}

void TunerCore::SetSearcher(const KernelId id, std::unique_ptr<Searcher> searcher)
{
    m_TuningRunner->SetSearcher(id, std::move(searcher));
}

void TunerCore::InitializeConfigurationData(const KernelId id)
{
    const auto& kernel = m_KernelManager->GetKernel(id);
    m_TuningRunner->InitializeConfigurationData(kernel);
}

void TunerCore::ClearConfigurationData(const KernelId id)
{
    m_TuningRunner->ClearConfigurationData(id);
}

uint64_t TunerCore::GetConfigurationsCount(const KernelId id) const
{
    return m_TuningRunner->GetConfigurationsCount(id);
}

KernelConfiguration TunerCore::GetBestConfiguration(const KernelId id) const
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
        throw KttException("Kernel " + kernel.GetName() + " has multiple definitions, its source cannot be directly retrieved");
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
    TimeConfiguration::GetInstance().SetTimeUnit(unit);
}

void TunerCore::SaveResults(const std::vector<KernelResult>& results, const std::string& filePath, const OutputFormat format,
    const UserData& data) const
{
    if (results.empty())
    {
        throw KttException("Unable to save results because input vector is empty");
    }

    const std::string file = filePath + GetFileExtension(format);
    Logger::LogInfo("Saving kernel results to file: " + file);
    std::ofstream outputStream(file);

    if (!outputStream.is_open())
    {
        throw KttException("Unable to open file: " + file);
    }

    TunerMetadata metadata(*m_ComputeEngine);
    auto serializer = CreateSerializer(format);
    serializer->SerializeResults(metadata, results, data, outputStream);
}

std::vector<KernelResult> TunerCore::LoadResults(const std::string& filePath, const OutputFormat format, UserData& data) const
{
    const std::string file = filePath + GetFileExtension(format);
    Logger::LogInfo("Loading kernel results from file: " + file);
    std::ifstream inputStream(file);

    if (!inputStream.is_open())
    {
        throw KttException("Unable to open file: " + file);
    }

    auto deserializer = CreateDeserializer(format);
    const auto pair = deserializer->DeserializeResults(data, inputStream);

    if (pair.first.GetTimeUnit() != TimeConfiguration::GetInstance().GetTimeUnit())
    {
        Logger::LogWarning("Loaded kernel results use different time unit than tuner");
    }

    return pair.second;
}

QueueId TunerCore::AddComputeQueue(ComputeQueue queue)
{
    return m_ComputeEngine->AddComputeQueue(queue);
}

void TunerCore::RemoveComputeQueue(const QueueId id)
{
    m_ComputeEngine->RemoveComputeQueue(id);
}

void TunerCore::WaitForComputeAction(const ComputeActionId id)
{
    m_ComputeEngine->WaitForComputeAction(id);
}

void TunerCore::WaitForTransferAction(const TransferActionId id)
{
    m_ComputeEngine->WaitForTransferAction(id);
}

void TunerCore::SynchronizeQueue(const QueueId queueId)
{
    m_ComputeEngine->SynchronizeQueue(queueId);
}

void TunerCore::SynchronizeQueues()
{
    m_ComputeEngine->SynchronizeQueues();
}

void TunerCore::SynchronizeDevice()
{
    m_ComputeEngine->SynchronizeDevice();
}

void TunerCore::SetProfilingCounters(const std::vector<std::string>& counters)
{
    m_ComputeEngine->SetProfilingCounters(counters);
}

void TunerCore::SetCompilerOptions(const std::string& options, const bool overrideDefault)
{
    m_ComputeEngine->SetCompilerOptions(options, overrideDefault);
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

LoggingLevel TunerCore::GetLoggingLevel()
{
    return Logger::GetLogger().GetLoggingLevel();
}

void TunerCore::Log(const LoggingLevel level, const std::string& message)
{
    Logger::GetLogger().Log(level, message);
}

void TunerCore::InitializeComputeEngine([[maybe_unused]] const PlatformIndex platform, const DeviceIndex device, const ComputeApi api,
    const uint32_t queueCount)
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
        m_ComputeEngine = std::make_unique<VulkanEngine>(device, queueCount);
        #else
        throw KttException("Support for Vulkan API is not included in this version of KTT framework");
        #endif // KTT_API_VULKAN
        break;
    default:
        KttError("Unhandled compute API value");
    }
}

void TunerCore::InitializeComputeEngine(const ComputeApi api, [[maybe_unused]] const ComputeApiInitializer& initializer,
    [[maybe_unused]] std::vector<QueueId>& assignedQueueIds)
{
    switch (api)
    {
    case ComputeApi::OpenCL:
        #ifdef KTT_API_OPENCL
        m_ComputeEngine = std::make_unique<OpenClEngine>(initializer, assignedQueueIds);
        #else
        throw KttException("Support for OpenCL API is not included in this version of KTT framework");
        #endif // KTT_API_OPENCL
        break;
    case ComputeApi::CUDA:
        #ifdef KTT_API_CUDA
        m_ComputeEngine = std::make_unique<CudaEngine>(initializer, assignedQueueIds);
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
    m_TuningRunner = std::make_unique<TuningRunner>(*m_KernelRunner);
}

std::unique_ptr<Serializer> TunerCore::CreateSerializer(const OutputFormat format)
{
    switch (format)
    {
    case OutputFormat::JSON:
        return std::make_unique<JsonSerializer>();
    case OutputFormat::XML:
        return std::make_unique<XmlSerializer>();
    default:
        KttError("Unhandled output format value");
        return nullptr;
    }
}

std::unique_ptr<Deserializer> TunerCore::CreateDeserializer(const OutputFormat format)
{
    switch (format)
    {
    case OutputFormat::JSON:
        return std::make_unique<JsonDeserializer>();
    case OutputFormat::XML:
        return std::make_unique<XmlDeserializer>();
    default:
        KttError("Unhandled output format value");
        return nullptr;
    }
}

} // namespace ktt
