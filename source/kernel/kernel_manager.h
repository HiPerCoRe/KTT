#pragma once

#include <vector>

#include "kernel.h"
#include "kernel_composition.h"
#include "kernel_configuration.h"
#include "api/device_info.h"
#include "enum/dimension_vector_type.h"
#include "enum/global_size_type.h"

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
    size_t addKernelComposition(const std::vector<size_t>& kernelIds);
    std::string getKernelSourceWithDefines(const size_t id, const KernelConfiguration& kernelConfiguration) const;
    KernelConfiguration getKernelConfiguration(const size_t id, const std::vector<ParameterValue>& parameterValues) const;
    std::vector<KernelConfiguration> getKernelConfigurations(const size_t id, const DeviceInfo& deviceInfo) const;
    std::vector<KernelConfiguration> getCompositionKernelConfigurations(const size_t compositionId, const DeviceInfo& deviceInfo) const;
    void setGlobalSizeType(const GlobalSizeType& globalSizeType);

    // Kernel modification methods
    void addParameter(const size_t id, const std::string& name, const std::vector<size_t>& values, const ThreadModifierType& threadModifierType,
        const ThreadModifierAction& threadModifierAction, const Dimension& modifierDimension);
    void addConstraint(const size_t id, const std::function<bool(std::vector<size_t>)>& constraintFunction,
        const std::vector<std::string>& parameterNames);
    void setArguments(const size_t id, const std::vector<size_t>& argumentIndices);
    void addCompositionKernelParameter(const size_t compositionId, const size_t kernelId, const std::string& parameterName,
        const std::vector<size_t>& parameterValues, const ThreadModifierType& threadModifierType, const ThreadModifierAction& threadModifierAction,
        const Dimension& modifierDimension);
    void setCompositionKernelArguments(const size_t compositionId, const size_t kernelId, const std::vector<size_t>& argumentIds);

    // Getters
    size_t getKernelCount() const;
    const Kernel& getKernel(const size_t id) const;
    Kernel& getKernel(const size_t id);
    size_t getCompositionCount() const;
    const KernelComposition& getKernelComposition(const size_t id) const;
    KernelComposition& getKernelComposition(const size_t id);
    bool isKernel(const size_t id) const;
    bool isKernelComposition(const size_t id) const;

private:
    // Attributes
    size_t nextId;
    std::vector<Kernel> kernels;
    std::vector<KernelComposition> kernelCompositions;
    GlobalSizeType globalSizeType;

    // Helper methods
    std::string loadFileToString(const std::string& filePath) const;
    void computeConfigurations(const size_t currentParameterIndex, const DeviceInfo& deviceInfo, const std::vector<KernelParameter>& parameters,
        const std::vector<KernelConstraint>& constraints, const std::vector<ParameterValue>& parameterValues, const DimensionVector& globalSize,
        const DimensionVector& localSize, std::vector<KernelConfiguration>& finalResult) const;
    void computeCompositionConfigurations(const size_t currentParameterIndex, const DeviceInfo& deviceInfo,
        const std::vector<KernelParameter>& parameters, const std::vector<KernelConstraint>& constraints,
        const std::vector<ParameterValue>& parameterValues, std::vector<std::pair<size_t, DimensionVector>>& globalSizes,
        std::vector<std::pair<size_t, DimensionVector>>& localSizes, std::vector<KernelConfiguration>& finalResult) const;
    DimensionVector modifyDimensionVector(const DimensionVector& vector, const DimensionVectorType& dimensionVectorType,
        const KernelParameter& parameter, const size_t parameterValue) const;
    bool configurationIsValid(const KernelConfiguration& configuration, const std::vector<KernelConstraint>& constraints,
        const DeviceInfo& deviceInfo) const;
};

} // namespace ktt
