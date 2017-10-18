#pragma once

#include <fstream>
#include <memory>
#include <vector>

#include "compute_engine/compute_engine.h"
#include "enum/compute_api.h"
#include "enum/run_mode.h"
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
    explicit TunerCore(const size_t platformIndex, const size_t deviceIndex, const ComputeApi& computeApi, const RunMode& runMode);

    // Kernel manager methods
    size_t addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize, const DimensionVector& localSize);
    size_t addKernelFromFile(const std::string& filePath, const std::string& kernelName, const DimensionVector& globalSize,
        const DimensionVector& localSize);
    size_t addKernelComposition(const std::string& compositionName, const std::vector<size_t>& kernelIds,
        std::unique_ptr<TuningManipulator> tuningManipulator);
    void addParameter(const size_t kernelId, const std::string& parameterName, const std::vector<size_t>& parameterValues,
        const ThreadModifierType& threadModifierType, const ThreadModifierAction& threadModifierAction, const Dimension& modifierDimension);
    void addConstraint(const size_t kernelId, const std::function<bool(std::vector<size_t>)>& constraintFunction,
        const std::vector<std::string>& parameterNames);
    void setKernelArguments(const size_t kernelId, const std::vector<size_t>& argumentIndices);
    void addCompositionKernelParameter(const size_t compositionId, const size_t kernelId, const std::string& parameterName,
        const std::vector<size_t>& parameterValues, const ThreadModifierType& threadModifierType, const ThreadModifierAction& threadModifierAction,
        const Dimension& modifierDimension);
    void setCompositionKernelArguments(const size_t compositionId, const size_t kernelId, const std::vector<size_t>& argumentIds);
    void setGlobalSizeType(const GlobalSizeType& globalSizeType);

    // Argument manager methods
    size_t addArgument(const void* data, const size_t numberOfElements, const ArgumentDataType& dataType,
        const ArgumentMemoryLocation& memoryLocation, const ArgumentAccessType& accessType, const ArgumentUploadType& uploadType);

    // Tuning runner methods
    void tuneKernel(const size_t kernelId);
    void runKernel(const size_t kernelId, const std::vector<ParameterValue>& kernelConfiguration,
        const std::vector<ArgumentOutputDescriptor>& outputDescriptors);
    void setSearchMethod(const SearchMethod& searchMethod, const std::vector<double>& searchArguments);
    void setValidationMethod(const ValidationMethod& validationMethod, const double toleranceThreshold);
    void setValidationRange(const size_t argumentId, const size_t validationRange);
    void setReferenceKernel(const size_t kernelId, const size_t referenceKernelId, const std::vector<ParameterValue>& referenceKernelConfiguration,
        const std::vector<size_t>& resultArgumentIds);
    void setReferenceClass(const size_t kernelId, std::unique_ptr<ReferenceClass> referenceClass, const std::vector<size_t>& resultArgumentIds);
    void setTuningManipulator(const size_t kernelId, std::unique_ptr<TuningManipulator> tuningManipulator);
    void enableArgumentPrinting(const size_t argumentId, const std::string& filePath, const ArgumentPrintCondition& argumentPrintCondition);

    // Result printer methods
    void setPrintingTimeUnit(const TimeUnit& timeUnit);
    void setInvalidResultPrinting(const bool flag);
    void printResult(const size_t kernelId, std::ostream& outputTarget, const PrintFormat& printFormat) const;
    void printResult(const size_t kernelId, const std::string& filePath, const PrintFormat& printFormat) const;
    std::vector<ParameterValue> getBestConfiguration(const size_t kernelId) const;

    // Compute API methods
    void setCompilerOptions(const std::string& options);
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
