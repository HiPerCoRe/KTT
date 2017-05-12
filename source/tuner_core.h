#pragma once

#include <fstream>
#include <memory>
#include <vector>

#include "compute_api_driver/compute_api_driver.h"
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
    explicit TunerCore(const size_t platformIndex, const size_t deviceIndex);

    // Kernel manager methods
    size_t addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize, const DimensionVector& localSize);
    size_t addKernelFromFile(const std::string& filePath, const std::string& kernelName, const DimensionVector& globalSize,
        const DimensionVector& localSize);
    void addParameter(const size_t id, const std::string& name, const std::vector<size_t>& values, const ThreadModifierType& threadModifierType,
        const ThreadModifierAction& threadModifierAction, const Dimension& modifierDimension);
    void addConstraint(const size_t id, const std::function<bool(std::vector<size_t>)>& constraintFunction,
        const std::vector<std::string>& parameterNames);
    void setKernelArguments(const size_t id, const std::vector<size_t>& argumentIndices);
    void setSearchMethod(const size_t id, const SearchMethod& searchMethod, const std::vector<double>& searchArguments);
    void setReferenceKernel(const size_t kernelId, const size_t referenceKernelId, const std::vector<ParameterValue>& referenceKernelConfiguration,
        const std::vector<size_t>& resultArgumentIds);
    void setReferenceClass(const size_t kernelId, std::unique_ptr<ReferenceClass> referenceClass, const std::vector<size_t>& resultArgumentIds);
    void setTuningManipulator(const size_t kernelId, std::unique_ptr<TuningManipulator> tuningManipulator);

    // Argument manager methods
    size_t addArgument(const void* data, const size_t numberOfElements, const ArgumentDataType& argumentDataType,
        const ArgumentMemoryType& argumentMemoryType, const ArgumentUploadType& argumentUploadType);

    // Tuning runner methods
    void tuneKernel(const size_t id);
    void setValidationMethod(const ValidationMethod& validationMethod, const double toleranceThreshold);
    void enableArgumentPrinting(const size_t argumentId, const std::string& filePath, const ArgumentPrintCondition& argumentPrintCondition);

    // Result printer methods
    void setPrintingTimeUnit(const TimeUnit& timeUnit);
    void printResult(const size_t kernelId, std::ostream& outputTarget, const PrintFormat& printFormat) const;
    void printResult(const size_t kernelId, const std::string& filePath, const PrintFormat& printFormat) const;

    // Compute API methods
    void setCompilerOptions(const std::string& options);
    void printComputeApiInfo(std::ostream& outputTarget) const;
    std::vector<PlatformInfo> getPlatformInfo() const;
    std::vector<DeviceInfo> getDeviceInfo(const size_t platformIndex) const;

    // Logger methods
    void setLoggingTarget(std::ostream& outputTarget);
    void setLoggingTarget(const std::string& filePath);
    void log(const std::string& message) const;

private:
    // Attributes
    std::unique_ptr<ArgumentManager> argumentManager;
    std::unique_ptr<KernelManager> kernelManager;
    std::unique_ptr<ComputeApiDriver> computeApiDriver;
    std::unique_ptr<TuningRunner> tuningRunner;
    Logger logger;
    ResultPrinter resultPrinter;
};

} // namespace ktt
