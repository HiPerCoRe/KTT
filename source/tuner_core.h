#pragma once

#include <fstream>
#include <memory>
#include <vector>
#include "compute_engine/compute_engine.h"
#include "enum/compute_api.h"
#include "kernel/kernel_manager.h"
#include "kernel_argument/argument_manager.h"
#include "tuning_runner/tuning_runner.h"
#include "utility/logger.h"
#include "utility/result_printer.h"

namespace ktt
{

class TunerCore
{
public:
    // Constructor
    explicit TunerCore(const size_t platformIndex, const size_t deviceIndex, const ComputeApi& computeApi);

    // Kernel manager methods
    KernelId addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize, const DimensionVector& localSize);
    KernelId addKernelFromFile(const std::string& filePath, const std::string& kernelName, const DimensionVector& globalSize,
        const DimensionVector& localSize);
    KernelId addComposition(const std::string& compositionName, const std::vector<KernelId>& kernelIds,
        std::unique_ptr<TuningManipulator> manipulator);
    void addParameter(const KernelId id, const std::string& parameterName, const std::vector<size_t>& parameterValues,
        const ThreadModifierType& modifierType, const ThreadModifierAction& modifierAction, const Dimension& modifierDimension);
    void addParameter(const KernelId id, const std::string& parameterName, const std::vector<double>& parameterValues);
    void addConstraint(const KernelId id, const std::function<bool(std::vector<size_t>)>& constraintFunction,
        const std::vector<std::string>& parameterNames);
    void setKernelArguments(const KernelId id, const std::vector<ArgumentId>& argumentIds);
    void addCompositionKernelParameter(const KernelId compositionId, const KernelId kernelId, const std::string& parameterName,
        const std::vector<size_t>& parameterValues, const ThreadModifierType& modifierType, const ThreadModifierAction& modifierAction,
        const Dimension& modifierDimension);
    void setCompositionKernelArguments(const KernelId compositionId, const KernelId kernelId, const std::vector<ArgumentId>& argumentIds);

    // Argument manager methods
    ArgumentId addArgument(void* data, const size_t numberOfElements, const size_t elementSizeInBytes, const ArgumentDataType& dataType,
        const ArgumentMemoryLocation& memoryLocation, const ArgumentAccessType& accessType, const ArgumentUploadType& uploadType,
        const bool copyData);
    ArgumentId addArgument(const void* data, const size_t numberOfElements, const size_t elementSizeInBytes, const ArgumentDataType& dataType,
        const ArgumentMemoryLocation& memoryLocation, const ArgumentAccessType& accessType, const ArgumentUploadType& uploadType);

    // Tuning runner methods
    void tuneKernel(const KernelId id);
    void tuneKernelByStep(const KernelId id, const std::vector<ArgumentOutputDescriptor>& output);
    void runKernel(const KernelId id, const std::vector<ParameterPair>& configuration, const std::vector<ArgumentOutputDescriptor>& output);
    void setSearchMethod(const SearchMethod& method, const std::vector<double>& arguments);
    void setValidationMethod(const ValidationMethod& method, const double toleranceThreshold);
    void setValidationRange(const ArgumentId id, const size_t range);
    void setArgumentComparator(const ArgumentId id, const std::function<bool(const void*, const void*)>& comparator);
    void setReferenceKernel(const KernelId id, const KernelId referenceId, const std::vector<ParameterPair>& referenceConfiguration,
        const std::vector<ArgumentId>& validatedArgumentIds);
    void setReferenceClass(const KernelId id, std::unique_ptr<ReferenceClass> referenceClass, const std::vector<ArgumentId>& validatedArgumentIds);
    void setTuningManipulator(const KernelId id, std::unique_ptr<TuningManipulator> manipulator);
    std::vector<ParameterPair> getBestConfiguration(const KernelId id) const;

    // Result printer methods
    void setPrintingTimeUnit(const TimeUnit& unit);
    void setInvalidResultPrinting(const bool flag);
    void printResult(const KernelId id, std::ostream& outputTarget, const PrintFormat& format) const;
    void printResult(const KernelId id, const std::string& filePath, const PrintFormat& format) const;

    // Compute engine methods
    void setCompilerOptions(const std::string& options);
    void setGlobalSizeType(const GlobalSizeType& type);
    void setAutomaticGlobalSizeCorrection(const bool flag);
    void printComputeApiInfo(std::ostream& outputTarget) const;
    std::vector<PlatformInfo> getPlatformInfo() const;
    std::vector<DeviceInfo> getDeviceInfo(const size_t platformIndex) const;
    DeviceInfo getCurrentDeviceInfo() const;

    // Logger methods
    void setLoggingTarget(std::ostream& outputTarget);
    void setLoggingTarget(const std::string& filePath);
    void log(const std::string& message) const;

private:
    // Attributes
    std::unique_ptr<ArgumentManager> argumentManager;
    std::unique_ptr<KernelManager> kernelManager;
    std::unique_ptr<ComputeEngine> computeEngine;
    std::unique_ptr<TuningRunner> tuningRunner;
    Logger logger;
    ResultPrinter resultPrinter;
};

} // namespace ktt
