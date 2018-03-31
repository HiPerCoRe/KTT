#pragma once

#include <fstream>
#include <memory>
#include <vector>
#include "ktt_types.h"
#include "compute_engine/compute_engine.h"
#include "enum/compute_api.h"
#include "kernel/kernel_manager.h"
#include "kernel_argument/argument_manager.h"
#include "tuning_runner/kernel_runner.h"
#include "tuning_runner/tuning_runner.h"
#include "utility/logger.h"
#include "utility/result_printer.h"
#include "utility/result_loader.h"

namespace ktt
{

class TunerCore
{
public:
    // Constructor
    explicit TunerCore(const PlatformIndex platform, const DeviceIndex device, const ComputeAPI computeAPI, const uint32_t queueCount);

    // Kernel manager methods
    KernelId addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize,
        const DimensionVector& localSize);
    KernelId addKernelFromFile(const std::string& filePath, const std::string& kernelName, const DimensionVector& globalSize,
        const DimensionVector& localSize);
    KernelId addComposition(const std::string& compositionName, const std::vector<KernelId>& kernelIds,
        std::unique_ptr<TuningManipulator> manipulator);
    void addParameter(const KernelId id, const std::string& parameterName, const std::vector<size_t>& parameterValues,
        const ModifierType modifierType, const ModifierAction modifierAction, const ModifierDimension modifierDimension);
    void addParameter(const KernelId id, const std::string& parameterName, const std::vector<double>& parameterValues);
    void addLocalMemoryModifier(const KernelId id, const std::string& parameterName, const ArgumentId argumentId,
        const ModifierAction modifierAction);
    void addConstraint(const KernelId id, const std::function<bool(std::vector<size_t>)>& constraintFunction,
        const std::vector<std::string>& parameterNames);
    void setKernelArguments(const KernelId id, const std::vector<ArgumentId>& argumentIds);
    void addCompositionKernelParameter(const KernelId compositionId, const KernelId kernelId, const std::string& parameterName,
        const std::vector<size_t>& parameterValues, const ModifierType modifierType, const ModifierAction modifierAction,
        const ModifierDimension modifierDimension);
    void addCompositionKernelLocalMemoryModifier(const KernelId compositionId, const KernelId kernelId, const std::string& parameterName,
        const ArgumentId argumentId, const ModifierAction modifierAction);
    void setCompositionKernelArguments(const KernelId compositionId, const KernelId kernelId, const std::vector<ArgumentId>& argumentIds);
    std::string getKernelSource(const KernelId id, const std::vector<ParameterPair>& configuration) const;

    // Argument manager methods
    ArgumentId addArgument(void* data, const size_t numberOfElements, const size_t elementSizeInBytes, const ArgumentDataType dataType,
        const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const ArgumentUploadType uploadType,
        const bool copyData);
    ArgumentId addArgument(const void* data, const size_t numberOfElements, const size_t elementSizeInBytes, const ArgumentDataType dataType,
        const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const ArgumentUploadType uploadType);

    // Kernel runner methods
    bool runKernel(const KernelId id, const std::vector<ParameterPair>& configuration, const std::vector<OutputDescriptor>& output);
    void setTuningManipulator(const KernelId id, std::unique_ptr<TuningManipulator> manipulator);

    // Kernel tuner methods
    void tuneKernel(const KernelId id, std::unique_ptr<StopCondition> stopCondition);
    void dryTuneKernel(const KernelId id, const std::string& filePath);
    bool tuneKernelByStep(const KernelId id, const std::vector<OutputDescriptor>& output, const bool recomputeReference);
    void setSearchMethod(const SearchMethod method, const std::vector<double>& arguments);
    void setValidationMethod(const ValidationMethod method, const double toleranceThreshold);
    void setValidationRange(const ArgumentId id, const size_t range);
    void setArgumentComparator(const ArgumentId id, const std::function<bool(const void*, const void*)>& comparator);
    void setReferenceKernel(const KernelId id, const KernelId referenceId, const std::vector<ParameterPair>& referenceConfiguration,
        const std::vector<ArgumentId>& validatedArgumentIds);
    void setReferenceClass(const KernelId id, std::unique_ptr<ReferenceClass> referenceClass, const std::vector<ArgumentId>& validatedArgumentIds);
    std::pair<std::vector<ParameterPair>, double> getBestConfiguration(const KernelId id) const;

    // Result printer methods
    void setPrintingTimeUnit(const TimeUnit unit);
    void setInvalidResultPrinting(const bool flag);
    void printResult(const KernelId id, std::ostream& outputTarget, const PrintFormat format) const;
    void printResult(const KernelId id, const std::string& filePath, const PrintFormat format) const;

    // Compute engine methods
    void setCompilerOptions(const std::string& options);
    void setGlobalSizeType(const GlobalSizeType type);
    void setAutomaticGlobalSizeCorrection(const bool flag);
    void printComputeAPIInfo(std::ostream& outputTarget) const;
    std::vector<PlatformInfo> getPlatformInfo() const;
    std::vector<DeviceInfo> getDeviceInfo(const PlatformIndex platform) const;
    DeviceInfo getCurrentDeviceInfo() const;

    // Logger methods
    void setLoggingTarget(std::ostream& outputTarget);
    void setLoggingTarget(const std::string& filePath);
    void log(const std::string& message) const;

private:
    // Attributes
    Logger logger;
    ResultPrinter resultPrinter;
    std::unique_ptr<ArgumentManager> argumentManager;
    std::unique_ptr<KernelManager> kernelManager;
    std::unique_ptr<ComputeEngine> computeEngine;
    std::unique_ptr<KernelRunner> kernelRunner;
    std::unique_ptr<TuningRunner> tuningRunner;
};

} // namespace ktt
