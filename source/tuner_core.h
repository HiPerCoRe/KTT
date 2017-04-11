#pragma once

#include <memory>
#include <fstream>
#include <vector>

#include "compute_api_driver/opencl/opencl_core.h"
#include "kernel/kernel_manager.h"
#include "kernel_argument/argument_manager.h"
#include "tuning_runner/tuning_runner.h"
#include "result_printer.h"

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

    // Argument manager methods
    template <typename T> size_t addArgument(const std::vector<T>& data, const ArgumentMemoryType& argumentMemoryType,
        const ArgumentQuantity& argumentQuantity)
    {
        return argumentManager->addArgument(data, argumentMemoryType, argumentQuantity);
    }

    template <typename T> void updateArgument(const size_t id, const std::vector<T>& data, const ArgumentQuantity& argumentQuantity)
    {
        argumentManager->updateArgument(id, data, argumentQuantity);
    }

    // Tuning runner methods
    void tuneKernel(const size_t id);
    void setValidationMethod(const ValidationMethod& validationMethod, const double toleranceThreshold);

    // Result printer methods
    void printResult(const size_t kernelId, std::ostream& outputTarget, const PrintFormat& printFormat) const;
    void printResult(const size_t kernelId, const std::string& filePath, const PrintFormat& printFormat) const;

    // Compute API methods
    void setCompilerOptions(const std::string& options);
    static void printComputeAPIInfo(std::ostream& outputTarget);
    static std::vector<PlatformInfo> getPlatformInfo();
    static std::vector<DeviceInfo> getDeviceInfo(const size_t platformIndex);

private:
    // Attributes
    std::unique_ptr<ArgumentManager> argumentManager;
    std::unique_ptr<KernelManager> kernelManager;
    std::unique_ptr<OpenCLCore> openCLCore;
    std::unique_ptr<TuningRunner> tuningRunner;
    std::unique_ptr<ResultPrinter> resultPrinter;
};

} // namespace ktt
