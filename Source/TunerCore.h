#pragma once

#include <memory>
#include <ostream>
#include <string>

#include <Api/ComputeApiInitializer.h>
#include <ComputeEngine/ComputeApi.h>
#include <ComputeEngine/ComputeEngine.h>
#include <Kernel/KernelManager.h>
#include <KernelArgument/KernelArgumentManager.h>
#include <KernelRunner/KernelRunner.h>
#include <Output/Deserializer/Deserializer.h>
#include <Output/Serializer/Serializer.h>
#include <Output/OutputFormat.h>
#include <TuningRunner/TuningRunner.h>
#include <Utility/Logger/LoggingLevel.h>
#include <KttTypes.h>

namespace ktt
{

class TunerCore
{
public:
    explicit TunerCore(const PlatformIndex platform, const DeviceIndex device, const ComputeApi api, const uint32_t queueCount);
    explicit TunerCore(const ComputeApi api, const ComputeApiInitializer& initializer, std::vector<QueueId>& assignedQueueIds);

    // Kernel management
    KernelDefinitionId AddKernelDefinition(const std::string& name, const std::string& source, const DimensionVector& globalSize,
        const DimensionVector& localSize, const std::vector<std::string>& typeNames = {});
    KernelDefinitionId AddKernelDefinitionFromFile(const std::string& name, const std::string& filePath,
        const DimensionVector& globalSize, const DimensionVector& localSize, const std::vector<std::string>& typeNames = {});
    KernelDefinitionId GetKernelDefinitionId(const std::string& name, const std::vector<std::string>& typeNames = {}) const;
    void RemoveKernelDefinition(const KernelDefinitionId id);
    void SetArguments(const KernelDefinitionId id, const std::vector<ArgumentId>& argumentIds);
    KernelId CreateKernel(const std::string& name, const KernelDefinitionId definitionId);
    KernelId CreateKernel(const std::string& name, const std::vector<KernelDefinitionId>& definitionIds, KernelLauncher launcher);
    void RemoveKernel(const KernelId id);
    void SetLauncher(const KernelId id, KernelLauncher launcher);
    void AddParameter(const KernelId id, const std::string& name, const std::vector<uint64_t>& values, const std::string& group);
    void AddParameter(const KernelId id, const std::string& name, const std::vector<double>& values, const std::string& group);
    void AddConstraint(const KernelId id, const std::vector<std::string>& parameters, ConstraintFunction function);
    void AddThreadModifier(const KernelId id, const std::vector<KernelDefinitionId>& definitionIds, const ModifierType type,
        const ModifierDimension dimension, const std::vector<std::string>& parameters, ModifierFunction function);
    void SetProfiledDefinitions(const KernelId id, const std::vector<KernelDefinitionId>& definitionIds);

    // Argument management
    ArgumentId AddArgumentWithReferencedData(const size_t elementSize, const ArgumentDataType dataType,
        const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const ArgumentMemoryType memoryType,
        const ArgumentManagementType managementType, void* data, const size_t dataSize);
    ArgumentId AddArgumentWithOwnedData(const size_t elementSize, const ArgumentDataType dataType,
        const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const ArgumentMemoryType memoryType,
        const ArgumentManagementType managementType, const void* data, const size_t dataSize, const std::string& symbolName = "");
    ArgumentId AddUserArgument(ComputeBuffer buffer, const size_t elementSize, const ArgumentDataType dataType,
        const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const size_t dataSize);
    void RemoveArgument(const ArgumentId id);
    void SetReadOnlyArgumentCache(const bool flag);

    // Kernel running and validation
    KernelResult RunKernel(const KernelId id, const KernelConfiguration& configuration, const std::vector<BufferOutputDescriptor>& output);
    void SetProfiling(const bool flag);
    void SetValidationMethod(const ValidationMethod method, const double toleranceThreshold);
    void SetValidationMode(const ValidationMode mode);
    void SetValidationRange(const ArgumentId id, const size_t range);
    void SetValueComparator(const ArgumentId id, ValueComparator comparator);
    void SetReferenceComputation(const ArgumentId id, ReferenceComputation computation);
    void SetReferenceKernel(const ArgumentId id, const KernelId referenceId, const KernelConfiguration& configuration);

    // Kernel tuning and configurations
    std::vector<KernelResult> TuneKernel(const KernelId id, std::unique_ptr<StopCondition> stopCondition);
    KernelResult TuneKernelIteration(const KernelId id, const std::vector<BufferOutputDescriptor>& output, const bool recomputeReference);
    std::vector<KernelResult> SimulateKernelTuning(const KernelId id, const std::vector<KernelResult>& results, const uint64_t iterations);
    void SetSearcher(const KernelId id, std::unique_ptr<Searcher> searcher);
    void ClearData(const KernelId id);
    KernelConfiguration GetBestConfiguration(const KernelId id) const;
    KernelConfiguration CreateConfiguration(const KernelId id, const ParameterInput& parameters) const;
    std::string GetKernelSource(const KernelId id, const KernelConfiguration& configuration) const;
    std::string GetKernelDefinitionSource(const KernelDefinitionId id, const KernelConfiguration& configuration) const;

    // Result printing
    static void SetTimeUnit(const TimeUnit unit);
    void SaveResults(const std::vector<KernelResult>& results, const std::string& filePath, const OutputFormat format,
        const UserData& data) const;
    std::vector<KernelResult> LoadResults(const std::string& filePath, const OutputFormat format, UserData& data) const;

    // Compute engine
    QueueId AddComputeQueue(ComputeQueue queue);
    void RemoveComputeQueue(const QueueId id);
    void WaitForComputeAction(const ComputeActionId id);
    void WaitForTransferAction(const TransferActionId id);
    void SynchronizeQueue(const QueueId queueId);
    void SynchronizeQueues();
    void SynchronizeDevice();
    void SetProfilingCounters(const std::vector<std::string>& counters);
    void SetCompilerOptions(const std::string& options);
    void SetGlobalSizeType(const GlobalSizeType type);
    void SetAutomaticGlobalSizeCorrection(const bool flag);
    void SetKernelCacheCapacity(const uint64_t capacity);
    std::vector<PlatformInfo> GetPlatformInfo() const;
    std::vector<DeviceInfo> GetDeviceInfo(const PlatformIndex platform) const;
    DeviceInfo GetCurrentDeviceInfo() const;

    // Logging
    static void SetLoggingLevel(const LoggingLevel level);
    static void SetLoggingTarget(std::ostream& target);
    static void SetLoggingTarget(const std::string& file);
    static void Log(const LoggingLevel level, const std::string& message);

private:
    std::unique_ptr<KernelArgumentManager> m_ArgumentManager;
    std::unique_ptr<KernelManager> m_KernelManager;
    std::unique_ptr<ComputeEngine> m_ComputeEngine;
    std::unique_ptr<KernelRunner> m_KernelRunner;
    std::unique_ptr<TuningRunner> m_TuningRunner;

    void InitializeComputeEngine(const PlatformIndex platform, const DeviceIndex device, const ComputeApi api, const uint32_t queueCount);
    void InitializeComputeEngine(const ComputeApi api, const ComputeApiInitializer& initializer, std::vector<QueueId>& assignedQueueIds);
    void InitializeRunners();

    static std::unique_ptr<Serializer> CreateSerializer(const OutputFormat format);
    static std::unique_ptr<Deserializer> CreateDeserializer(const OutputFormat format);
};

} // namespace ktt
