#pragma once

#include <vector>

#include "../dto/device_info.h"
#include "../enum/dimension_vector_type.h"
#include "kernel.h"
#include "kernel_configuration.h"

namespace ktt
{

class KernelManager
{
public:
    // Constructor
    KernelManager();

    // Core methods
    size_t addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize, const DimensionVector& localSize);
    size_t addKernelFromFile(const std::string& filePath, const std::string& kernelName, const DimensionVector& globalSize,
        const DimensionVector& localSize);
    std::string getKernelSourceWithDefines(const size_t id, const KernelConfiguration& kernelConfiguration) const;
    KernelConfiguration getKernelConfiguration(const size_t id, const std::vector<ParameterValue>& parameterValues) const;
    std::vector<KernelConfiguration> getKernelConfigurations(const size_t id, const DeviceInfo& deviceInfo) const;

    // Kernel modification methods
    void addParameter(const size_t id, const std::string& name, const std::vector<size_t>& values, const ThreadModifierType& threadModifierType,
        const ThreadModifierAction& threadModifierAction, const Dimension& modifierDimension);
    void addConstraint(const size_t id, const std::function<bool(std::vector<size_t>)>& constraintFunction,
        const std::vector<std::string>& parameterNames);
    void setArguments(const size_t id, const std::vector<size_t>& argumentIndices);
    void setSearchMethod(const size_t id, const SearchMethod& searchMethod, const std::vector<double>& searchArguments);

    // Getters
    size_t getKernelCount() const;
    const Kernel* getKernel(const size_t id) const;
    Kernel* getKernel(const size_t id);

private:
    // Attributes
    size_t kernelCount;
    std::vector<Kernel> kernels;

    // Helper methods
    std::string loadFileToString(const std::string& filePath) const;
    void computeConfigurations(const size_t currentParameterIndex, const DeviceInfo& deviceInfo, const std::vector<KernelParameter>& parameters,
        const std::vector<KernelConstraint>& constraints, const std::vector<ParameterValue>& parameterValues, const DimensionVector& globalSize,
        const DimensionVector& localSize, std::vector<KernelConfiguration>& finalResult) const;
    DimensionVector modifyDimensionVector(const DimensionVector& vector, const DimensionVectorType& dimensionVectorType,
        const KernelParameter& parameter, const size_t parameterValue) const;
    bool configurationIsValid(const KernelConfiguration& configuration, const std::vector<KernelConstraint>& constraints,
        const DeviceInfo& deviceInfo) const;
};

} // namespace ktt
