#ifdef KTT_PYTHON
#include <pybind11/stl.h>
#endif // KTT_PYTHON

#include <Api/KttException.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/FileSystem.h>
#include <Tuner.h>
#include <TunerCore.h>

#ifdef KTT_PYTHON
#include <Api/Searcher/ProfileBasedSearcher.h>
#include <Python/PythonInterpreter.h>
#endif // KTT_PYTHON

namespace ktt
{

Tuner::Tuner(const PlatformIndex platform, const DeviceIndex device, const ComputeApi api) :
    m_Tuner(std::make_unique<TunerCore>(platform, device, api, 1))
{}

Tuner::Tuner(const PlatformIndex platform, const DeviceIndex device, const ComputeApi api, const uint32_t computeQueueCount) :
    m_Tuner(std::make_unique<TunerCore>(platform, device, api, computeQueueCount))
{}

Tuner::Tuner(const ComputeApi api, const ComputeApiInitializer& initializer)
{
    std::vector<QueueId> ids;
    m_Tuner = std::make_unique<TunerCore>(api, initializer, ids);
}

Tuner::Tuner(const ComputeApi api, const ComputeApiInitializer& initializer, std::vector<QueueId>& assignedQueueIds) :
    m_Tuner(std::make_unique<TunerCore>(api, initializer, assignedQueueIds))
{}

Tuner::~Tuner() = default;

KernelDefinitionId Tuner::AddKernelDefinition(const std::string& name, const std::string& source, const DimensionVector& globalSize,
    const DimensionVector& localSize, const std::vector<std::string>& typeNames)
{
    try
    {
        return m_Tuner->AddKernelDefinition(name, source, globalSize, localSize, typeNames);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
        return InvalidKernelDefinitionId;
    }
}

KernelDefinitionId Tuner::AddKernelDefinition(const std::string& name, const std::string& source,
    const std::vector<std::string>& typeNames)
{
    return AddKernelDefinition(name, source, DimensionVector(), DimensionVector(), typeNames);
}

KernelDefinitionId Tuner::AddKernelDefinitionFromFile(const std::string& name, const std::string& filePath,
    const DimensionVector& globalSize, const DimensionVector& localSize, const std::vector<std::string>& typeNames)
{
    try
    {
        return m_Tuner->AddKernelDefinitionFromFile(name, filePath, globalSize, localSize, typeNames);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
        return InvalidKernelDefinitionId;
    }
}

KernelDefinitionId Tuner::AddKernelDefinitionFromFile(const std::string& name, const std::string& filePath,
    const std::vector<std::string>& typeNames)
{
    return AddKernelDefinitionFromFile(name, filePath, DimensionVector(), DimensionVector(), typeNames);
}

KernelDefinitionId Tuner::GetKernelDefinitionId(const std::string& name, const std::vector<std::string>& typeNames) const
{
    try
    {
        return m_Tuner->GetKernelDefinitionId(name, typeNames);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
        return InvalidKernelDefinitionId;
    }
}

void Tuner::RemoveKernelDefinition(const KernelDefinitionId id)
{
    try
    {
        m_Tuner->RemoveKernelDefinition(id);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
    }
}

void Tuner::SetArguments(const KernelDefinitionId id, const std::vector<ArgumentId>& argumentIds)
{
    try
    {
        m_Tuner->SetArguments(id, argumentIds);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
    }
}

KernelId Tuner::CreateSimpleKernel(const std::string& name, const KernelDefinitionId definitionId)
{
    try
    {
        return m_Tuner->CreateKernel(name, definitionId);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
        return InvalidKernelId;
    }
}

KernelId Tuner::CreateCompositeKernel(const std::string& name, const std::vector<KernelDefinitionId>& definitionIds,
    KernelLauncher launcher)
{
    try
    {
        return m_Tuner->CreateKernel(name, definitionIds, launcher);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
        return InvalidKernelId;
    }
}

void Tuner::RemoveKernel(const KernelId id)
{
    try
    {
        m_Tuner->RemoveKernel(id);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
    }
}

void Tuner::SetLauncher(const KernelId id, KernelLauncher launcher)
{
    try
    {
        m_Tuner->SetLauncher(id, launcher);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
    }
}

void Tuner::AddScriptParameter(const KernelId id, const std::string& name, const ParameterValueType valueType, const std::string& valueScript,
    const std::string& group)
{
    try
    {
        m_Tuner->AddScriptParameter(id, name, valueType, valueScript, group);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
    }
}

void Tuner::AddThreadModifier(const KernelId id, const std::vector<KernelDefinitionId>& definitionIds, const ModifierType type,
    const ModifierDimension dimension, const std::vector<std::string>& parameters, ModifierFunction function)
{
    try
    {
        m_Tuner->AddThreadModifier(id, definitionIds, type, dimension, parameters, function);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
    }
}

void Tuner::AddThreadModifier(const KernelId id, const std::vector<KernelDefinitionId>& definitionIds, const ModifierType type,
    const ModifierDimension dimension, const std::string& parameter, const ModifierAction action)
{
    std::function<uint64_t(const uint64_t, const std::vector<uint64_t>&)> function;

    switch (action)
    {
    case ModifierAction::Add:
        function = [](const uint64_t size, const std::vector<uint64_t>& parameters)
        {
            return size + parameters[0];
        };
        break;
    case ModifierAction::Subtract:
        function = [](const uint64_t size, const std::vector<uint64_t>& parameters)
        {
            return size - parameters[0];
        };
        break;
    case ModifierAction::Multiply:
        function = [](const uint64_t size, const std::vector<uint64_t>& parameters)
        {
            return size * parameters[0];
        };
        break;
    case ModifierAction::Divide:
        function = [](const uint64_t size, const std::vector<uint64_t>& parameters)
        {
            return size / parameters[0];
        };
        break;
    case ModifierAction::DivideCeil:
        function = [](const uint64_t size, const std::vector<uint64_t>& parameters)
        {
            return (size + parameters[0] - 1) / parameters[0];
        };
        break;
    default:
        KttError("Unhandled modifier action value");
    }

    AddThreadModifier(id, definitionIds, type, dimension, std::vector<std::string>{parameter}, function);
}

void Tuner::AddScriptThreadModifier(const KernelId id, const std::vector<KernelDefinitionId>& definitionIds, const ModifierType type,
    const ModifierDimension dimension, const std::string& script)
{
    try
    {
        m_Tuner->AddScriptThreadModifier(id, definitionIds, type, dimension, script);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
    }
}

void Tuner::AddConstraint(const KernelId id, const std::vector<std::string>& parameters, ConstraintFunction function)
{
    try
    {
        m_Tuner->AddConstraint(id, parameters, function);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
    }
}

void Tuner::AddGenericConstraint(const KernelId id, const std::vector<std::string>& parameters, GenericConstraintFunction function)
{
    try
    {
        m_Tuner->AddGenericConstraint(id, parameters, function);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
    }
}

void Tuner::AddScriptConstraint(const KernelId id, const std::vector<std::string>& parameters, const std::string& script)
{
    try
    {
        m_Tuner->AddScriptConstraint(id, parameters, script);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
    }
}

void Tuner::SetProfiledDefinitions(const KernelId id, const std::vector<KernelDefinitionId>& definitionIds)
{
    try
    {
        m_Tuner->SetProfiledDefinitions(id, definitionIds);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
    }
}

ArgumentId Tuner::AddArgumentVector(ComputeBuffer buffer, const size_t bufferSize, const size_t elementSize,
    const ArgumentAccessType accessType, const ArgumentMemoryLocation memoryLocation)
{
    try
    {
        return AddUserArgument(buffer, elementSize, ArgumentDataType::Custom, memoryLocation, accessType, bufferSize);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
        return InvalidArgumentId;
    }
}

ArgumentId Tuner::AddArgumentVectorFromFile(const std::string& filePath, const ArgumentDataType dataType, const size_t elementSize,
    const ArgumentAccessType accessType, const ArgumentMemoryLocation memoryLocation, const ArgumentManagementType managementType)
{
    try
    {
        const auto data = LoadFileToBinary(filePath);
        return AddArgumentWithOwnedData(elementSize, dataType, memoryLocation, accessType, ArgumentMemoryType::Vector,
            managementType, data.data(), data.size());
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
        return InvalidArgumentId;
    }
}

void Tuner::SaveArgumentVector(const ArgumentId id, const std::string& filePath)
{
    try
    {
        m_Tuner->SaveArgument(id, filePath);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
    }
}

ArgumentId Tuner::AddArgumentScalar(const void* data, const size_t dataSize)
{
    try
    {
        return AddArgumentWithOwnedData(dataSize, ArgumentDataType::Custom, ArgumentMemoryLocation::Undefined,
            ArgumentAccessType::ReadOnly, ArgumentMemoryType::Scalar, ArgumentManagementType::Framework, data, dataSize);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
        return InvalidArgumentId;
    }
}

void Tuner::RemoveArgument(const ArgumentId id)
{
    try
    {
        m_Tuner->RemoveArgument(id);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
    }
}

void Tuner::SetReadOnlyArgumentCache(const bool flag)
{
    try
    {
        m_Tuner->SetReadOnlyArgumentCache(flag);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
    }
}

KernelResult Tuner::Run(const KernelId id, const KernelConfiguration& configuration,
    const std::vector<BufferOutputDescriptor>& output)
{
    return Run(id, configuration, {}, output);
}

KernelResult Tuner::Run(const KernelId id, const KernelConfiguration& configuration, const KernelDimensions& dimensions,
    const std::vector<BufferOutputDescriptor>& output)
{
    try
    {
        return m_Tuner->RunKernel(id, configuration, dimensions, output);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
        return KernelResult();
    }
}

void Tuner::SetProfiling(const bool flag)
{
    try
    {
        m_Tuner->SetProfiling(flag);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
    }
}

void Tuner::SetValidationMethod(const ValidationMethod method, const double toleranceThreshold)
{
    try
    {
        m_Tuner->SetValidationMethod(method, toleranceThreshold);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
    }
}

void Tuner::SetValidationMode(const ValidationMode mode)
{
    try
    {
        m_Tuner->SetValidationMode(mode);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
    }
}

void Tuner::SetValidationRange(const ArgumentId id, const size_t range)
{
    try
    {
        m_Tuner->SetValidationRange(id, range);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
    }
}

void Tuner::SetValueComparator(const ArgumentId id, ValueComparator comparator)
{
    try
    {
        m_Tuner->SetValueComparator(id, comparator);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
    }
}

void Tuner::SetReferenceComputation(const ArgumentId id, ReferenceComputation computation)
{
    try
    {
        m_Tuner->SetReferenceComputation(id, computation);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
    }
}

void Tuner::SetReferenceKernel(const ArgumentId id, const KernelId referenceId, const KernelConfiguration& configuration,
    const KernelDimensions& dimensions)
{
    try
    {
        m_Tuner->SetReferenceKernel(id, referenceId, configuration, dimensions);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
    }
}

std::vector<KernelResult> Tuner::Tune(const KernelId id, std::unique_ptr<StopCondition> stopCondition)
{
    return Tune(id, {}, std::move(stopCondition));
}

std::vector<KernelResult> Tuner::Tune(const KernelId id, const KernelDimensions& dimensions,
    std::unique_ptr<StopCondition> stopCondition)
{
    try
    {
        return m_Tuner->TuneKernel(id, dimensions, std::move(stopCondition));
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
        return std::vector<KernelResult>{};
    }
}

KernelResult Tuner::TuneIteration(const KernelId id, const std::vector<BufferOutputDescriptor>& output,
    const bool recomputeReference)
{
    return TuneIteration(id, {}, output, recomputeReference);
}

KernelResult Tuner::TuneIteration(const KernelId id, const KernelDimensions& dimensions,
    const std::vector<BufferOutputDescriptor>& output, const bool recomputeReference)
{
    try
    {
        return m_Tuner->TuneKernelIteration(id, dimensions, output, recomputeReference);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
        return KernelResult();
    }
}

std::vector<KernelResult> Tuner::SimulateKernelTuning(const KernelId id, const std::vector<KernelResult>& results,
    const uint64_t iterations)
{
    try
    {
        return m_Tuner->SimulateKernelTuning(id, results, iterations);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
        return std::vector<KernelResult>{};
    }
}

void Tuner::SetSearcher(const KernelId id, std::unique_ptr<Searcher> searcher)
{
    try
    {
        m_Tuner->SetSearcher(id, std::move(searcher));
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
    }
}

void Tuner::SetProfileBasedSearcher([[maybe_unused]] const KernelId id, [[maybe_unused]] const std::string& modelPath,
    [[maybe_unused]] const bool useBuiltinModule)
{
    try
    {
        #ifndef KTT_PYTHON
        throw KttException("Usage of profile-based searcher requires compilation of Python backend");
        #else
        if (useBuiltinModule)
        {
            SaveStringToFile(ProfileBasedSearcherFile, ProfileBasedSearcherModule);
        }

        PythonInterpreter::GetInterpreter();
        pybind11::module_ searcher = pybind11::module_::import(ProfileBasedSearcherName.c_str());
        searcher.attr("executeSearcher")(this, id, modelPath);
        #endif // KTT_PYTHON
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
    }
    #ifdef KTT_PYTHON
    catch (const pybind11::error_already_set& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
    }
    #endif // KTT_PYTHON
}

void Tuner::InitializeConfigurationData(const KernelId id)
{
    try
    {
        m_Tuner->InitializeConfigurationData(id);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
    }
}

void Tuner::ClearConfigurationData(const KernelId id)
{
    try
    {
        m_Tuner->ClearConfigurationData(id);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
    }
}

void Tuner::ClearData(const KernelId id)
{
    ClearConfigurationData(id);
}

uint64_t Tuner::GetConfigurationsCount(const KernelId id) const
{
    try
    {
        return m_Tuner->GetConfigurationsCount(id);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
        return 0;
    }
}

KernelConfiguration Tuner::GetBestConfiguration(const KernelId id) const
{
    try
    {
        return m_Tuner->GetBestConfiguration(id);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());

        static KernelConfiguration invalidConfiguration;
        return invalidConfiguration;
    }
}

KernelConfiguration Tuner::CreateConfiguration(const KernelId id, const ParameterInput& parameters) const
{
    try
    {
        return m_Tuner->CreateConfiguration(id, parameters);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
        return KernelConfiguration();
    }
}

std::string Tuner::GetKernelSource(const KernelId id, const KernelConfiguration& configuration) const
{
    try
    {
        return m_Tuner->GetKernelSource(id, configuration);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
        return "";
    }
}

std::string Tuner::GetKernelDefinitionSource(const KernelDefinitionId id, const KernelConfiguration& configuration) const
{
    try
    {
        return m_Tuner->GetKernelDefinitionSource(id, configuration);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
        return "";
    }
}

void Tuner::SetTimeUnit(const TimeUnit unit)
{
    TunerCore::SetTimeUnit(unit);
}

void Tuner::SaveResults(const std::vector<KernelResult>& results, const std::string& filePath, const OutputFormat format,
    const UserData& data) const
{
    try
    {
        m_Tuner->SaveResults(results, filePath, format, data);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
    }
}

std::vector<KernelResult> Tuner::LoadResults(const std::string& filePath, const OutputFormat format) const
{
    [[maybe_unused]] UserData emptyData;
    return LoadResults(filePath, format, emptyData);
}

std::vector<KernelResult> Tuner::LoadResults(const std::string& filePath, const OutputFormat format, UserData& data) const
{
    try
    {
        return m_Tuner->LoadResults(filePath, format, data);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
        return std::vector<KernelResult>{};
    }
}

QueueId Tuner::AddComputeQueue(ComputeQueue queue)
{
    try
    {
        return m_Tuner->AddComputeQueue(queue);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
        return InvalidQueueId;
    }
}

void Tuner::RemoveComputeQueue(const QueueId id)
{
    try
    {
        m_Tuner->RemoveComputeQueue(id);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
    }
}

void Tuner::WaitForComputeAction(const ComputeActionId id)
{
    try
    {
        m_Tuner->WaitForComputeAction(id);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
    }
}

void Tuner::WaitForTransferAction(const TransferActionId id)
{
    try
    {
        m_Tuner->WaitForTransferAction(id);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
    }
}

void Tuner::SynchronizeQueue(const QueueId id)
{
    try
    {
        m_Tuner->SynchronizeQueue(id);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
    }
}

void Tuner::SynchronizeQueues()
{
    try
    {
        m_Tuner->SynchronizeQueues();
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
    }
}

void Tuner::SynchronizeDevice()
{
    try
    {
        m_Tuner->SynchronizeDevice();
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
    }
}

void Tuner::Synchronize()
{
    SynchronizeDevice();
}

void Tuner::SetProfilingCounters(const std::vector<std::string>& counters)
{
    try
    {
        m_Tuner->SetProfilingCounters(counters);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
    }
}

void Tuner::SetCompilerOptions(const std::string& options, const bool overrideDefault)
{
    try
    {
        m_Tuner->SetCompilerOptions(options, overrideDefault);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
    }
}

void Tuner::SetGlobalSizeType(const GlobalSizeType type)
{
    try
    {
        m_Tuner->SetGlobalSizeType(type);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
    }
}

void Tuner::SetAutomaticGlobalSizeCorrection(const bool flag)
{
    try
    {
        m_Tuner->SetAutomaticGlobalSizeCorrection(flag);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
    }
}

void Tuner::SetKernelCacheCapacity(const uint64_t capacity)
{
    try
    {
        m_Tuner->SetKernelCacheCapacity(capacity);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
    }
}

std::vector<PlatformInfo> Tuner::GetPlatformInfo() const
{
    try
    {
        return m_Tuner->GetPlatformInfo();
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
        return std::vector<PlatformInfo>{};
    }
}

std::vector<DeviceInfo> Tuner::GetDeviceInfo(const PlatformIndex platform) const
{
    try
    {
        return m_Tuner->GetDeviceInfo(platform);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
        return std::vector<DeviceInfo>{};
    }
}

DeviceInfo Tuner::GetCurrentDeviceInfo() const
{
    try
    {
        return m_Tuner->GetCurrentDeviceInfo();
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
        return DeviceInfo(0, "");
    }
}

void Tuner::SetLoggingLevel(const LoggingLevel level)
{
    TunerCore::SetLoggingLevel(level);
}

LoggingLevel Tuner::GetLoggingLevel()
{
    return TunerCore::GetLoggingLevel();
}

void Tuner::SetLoggingTarget(std::ostream& outputTarget)
{
    TunerCore::SetLoggingTarget(outputTarget);
}

void Tuner::SetLoggingTarget(const std::string& filePath)
{
    TunerCore::SetLoggingTarget(filePath);
}

ArgumentId Tuner::AddArgumentWithReferencedData(const size_t elementSize, const ArgumentDataType dataType,
    const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const ArgumentMemoryType memoryType,
    const ArgumentManagementType managementType, void* data, const size_t dataSize)
{
    try
    {
        return m_Tuner->AddArgumentWithReferencedData(elementSize, dataType, memoryLocation, accessType, memoryType, managementType,
            data, dataSize);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
        return InvalidArgumentId;
    }
}

ArgumentId Tuner::AddArgumentWithOwnedData(const size_t elementSize, const ArgumentDataType dataType,
    const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const ArgumentMemoryType memoryType,
    const ArgumentManagementType managementType, const void* data, const size_t dataSize, const std::string& symbolName)
{
    try
    {
        return m_Tuner->AddArgumentWithOwnedData(elementSize, dataType, memoryLocation, accessType, memoryType, managementType,
            data, dataSize, symbolName);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
        return InvalidArgumentId;
    }
}

ArgumentId Tuner::AddUserArgument(ComputeBuffer buffer, const size_t elementSize, const ArgumentDataType dataType,
    const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const size_t dataSize)
{
    try
    {
        return m_Tuner->AddUserArgument(buffer, elementSize, dataType, memoryLocation, accessType, dataSize);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
        return InvalidArgumentId;
    }
}

void Tuner::AddParameterInternal(const KernelId id, const std::string& name, const std::vector<ParameterValue>& values,
    const std::string& group)
{
    try
    {
        m_Tuner->AddParameter(id, name, values, group);
    }
    catch (const KttException& exception)
    {
        TunerCore::Log(LoggingLevel::Error, exception.what());
    }
}

} // namespace ktt
