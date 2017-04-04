#pragma once

#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

// Type aliases and enums relevant to usage of API methods
#include "ktt_type_aliases.h"
#include "enum/dimension.h"
#include "enum/argument_memory_type.h"
#include "enum/print_format.h"
#include "enum/search_method.h"
#include "enum/thread_modifier_action.h"
#include "enum/thread_modifier_type.h"
#include "enum/validation_method.h"

// Information about platforms and devices
#include "dto/device_info.h"
#include "dto/platform_info.h"

// Reference class interface
#include "reference_class.h"

namespace ktt
{

class TunerCore; // Forward declaration of TunerCore class

class Tuner
{
public:
    // Constructor and destructor
    explicit Tuner(const size_t platformIndex, const size_t deviceIndex);
    ~Tuner();

    // Kernel handling methods
    size_t addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize, const DimensionVector& localSize);
    size_t addKernelFromFile(const std::string& filePath, const std::string& kernelName, const DimensionVector& globalSize,
        const DimensionVector& localSize);
    void addParameter(const size_t kernelId, const std::string& name, const std::vector<size_t>& values);
    void addParameter(const size_t kernelId, const std::string& name, const std::vector<size_t>& values,
        const ThreadModifierType& threadModifierType, const ThreadModifierAction& threadModifierAction, const Dimension& modifierDimension);
    void addConstraint(const size_t kernelId, const std::function<bool(std::vector<size_t>)>& constraintFunction,
        const std::vector<std::string>& parameterNames);
    void setKernelArguments(const size_t kernelId, const std::vector<size_t>& argumentIds);
    void setSearchMethod(const size_t kernelId, const SearchMethod& searchMethod, const std::vector<double>& searchArguments);

    // Argument handling methods
    template <typename T> size_t addArgument(const std::vector<T>& data, const ArgumentMemoryType& argumentMemoryType);
    template <typename T> size_t addArgument(const T value);
    template <typename T> void updateArgument(const size_t argumentId, const std::vector<T>& data);
    template <typename T> void updateArgument(const size_t argumentId, const T value);

    // Kernel tuning methods
    void tuneKernel(const size_t kernelId);

    // Result printing methods
    void printResult(const size_t kernelId, std::ostream& outputTarget, const PrintFormat& printFormat) const;
    void printResult(const size_t kernelId, const std::string& filePath, const PrintFormat& printFormat) const;

    // (WIP) Result validation methods
    void setValidationMethod(const ValidationMethod& validationMethod, const double toleranceThreshold);
    void setReferenceKernel(const size_t kernelId, const size_t referenceKernelId, const std::vector<ParameterValue>& referenceKernelConfiguration);
    void setReferenceClass(const size_t kernelId, std::unique_ptr<ReferenceClass> referenceClass);

    // Compute API methods
    void setCompilerOptions(const std::string& options);
    static void printComputeAPIInfo(std::ostream& outputTarget);
    static PlatformInfo getPlatformInfo(const size_t platformIndex);
    static std::vector<PlatformInfo> getPlatformInfoAll();
    static DeviceInfo getDeviceInfo(const size_t platformIndex, const size_t deviceIndex);
    static std::vector<DeviceInfo> getDeviceInfoAll(const size_t platformIndex);

private:
    // Attributes
    std::unique_ptr<TunerCore> tunerCore;
};

} // namespace ktt
