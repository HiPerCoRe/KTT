#pragma once

#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

// Type aliases and enums relevant to usage of API methods
#include "ktt_type_aliases.h"
#include "enum/argument_memory_type.h"
#include "enum/argument_print_condition.h"
#include "enum/dimension.h"
#include "enum/print_format.h"
#include "enum/search_method.h"
#include "enum/thread_modifier_action.h"
#include "enum/thread_modifier_type.h"
#include "enum/validation_method.h"

// Information about platforms and devices
#include "dto/device_info.h"
#include "dto/platform_info.h"

// Reference class interface
#include "customization/reference_class.h"

// Tuning manipulator interface
#include "customization/tuning_manipulator.h"

namespace ktt
{

class TunerCore; // Forward declaration of TunerCore class

class Tuner
{
public:
    // Constructor and destructor
    explicit Tuner(const size_t platformIndex, const size_t deviceIndex);
    ~Tuner();

    // Basic kernel handling methods
    size_t addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize, const DimensionVector& localSize);
    size_t addKernelFromFile(const std::string& filePath, const std::string& kernelName, const DimensionVector& globalSize,
        const DimensionVector& localSize);
    void setKernelArguments(const size_t kernelId, const std::vector<size_t>& argumentIds);
    void addParameter(const size_t kernelId, const std::string& name, const std::vector<size_t>& values);

    // Advanced kernel handling methods
    void addParameter(const size_t kernelId, const std::string& name, const std::vector<size_t>& values,
        const ThreadModifierType& threadModifierType, const ThreadModifierAction& threadModifierAction, const Dimension& modifierDimension);
    void addConstraint(const size_t kernelId, const std::function<bool(std::vector<size_t>)>& constraintFunction,
        const std::vector<std::string>& parameterNames);
    void setSearchMethod(const size_t kernelId, const SearchMethod& searchMethod, const std::vector<double>& searchArguments);
    void setTuningManipulator(const size_t kernelId, std::unique_ptr<TuningManipulator> tuningManipulator);

    // Argument handling methods
    template <typename T> size_t addArgument(const std::vector<T>& data, const ArgumentMemoryType& argumentMemoryType);
    template <typename T> size_t addArgument(const T value);
    void enableArgumentPrinting(const std::vector<size_t> argumentIds, const std::string& filePath,
        const ArgumentPrintCondition& argumentPrintCondition);

    // Kernel tuning methods
    void tuneKernel(const size_t kernelId);

    // Result printing methods
    void printResult(const size_t kernelId, std::ostream& outputTarget, const PrintFormat& printFormat) const;
    void printResult(const size_t kernelId, const std::string& filePath, const PrintFormat& printFormat) const;

    // Result validation methods
    void setReferenceKernel(const size_t kernelId, const size_t referenceKernelId, const std::vector<ParameterValue>& referenceKernelConfiguration,
        const std::vector<size_t>& resultArgumentIds);
    void setReferenceClass(const size_t kernelId, std::unique_ptr<ReferenceClass> referenceClass, const std::vector<size_t>& resultArgumentIds);
    void setValidationMethod(const ValidationMethod& validationMethod, const double toleranceThreshold);

    // Compute API methods
    void setCompilerOptions(const std::string& options);
    static void printComputeAPIInfo(std::ostream& outputTarget);
    static std::vector<PlatformInfo> getPlatformInfo();
    static std::vector<DeviceInfo> getDeviceInfo(const size_t platformIndex);

    // Utility methods
    void setLoggingTarget(std::ostream& outputTarget);
    void setLoggingTarget(const std::string& filePath);

private:
    // Attributes
    std::unique_ptr<TunerCore> tunerCore;
};

} // namespace ktt
