#pragma once

#include <vector>
#include "kernel.h"
#include "kernel_composition.h"
#include "kernel_configuration.h"
#include "api/device_info.h"
#include "enum/dimension_vector_type.h"

namespace ktt
{

class KernelManager
{
public:
    // Constructor
    KernelManager(const DeviceInfo& currentDeviceInfo);

    // Core methods
    KernelId addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize, const DimensionVector& localSize);
    KernelId addKernelFromFile(const std::string& filePath, const std::string& kernelName, const DimensionVector& globalSize,
        const DimensionVector& localSize);
    KernelId addKernelComposition(const std::string& compositionName, const std::vector<KernelId>& kernelIds);
    std::string getKernelSourceWithDefines(const KernelId id, const KernelConfiguration& configuration) const;
    std::string getKernelSourceWithDefines(const KernelId id, const std::vector<ParameterPair>& configuration) const;
    KernelConfiguration getKernelConfiguration(const KernelId id, const std::vector<ParameterPair>& parameterPairs) const;
    KernelConfiguration getKernelCompositionConfiguration(const KernelId compositionId, const std::vector<ParameterPair>& parameterPairs) const;
    std::vector<KernelConfiguration> getKernelConfigurations(const KernelId id) const;
    std::vector<KernelConfiguration> getKernelCompositionConfigurations(const KernelId compositionId) const;

    // Kernel modification methods
    void addParameter(const KernelId id, const std::string& name, const std::vector<size_t>& values, const ModifierType& modifierType,
        const ModifierAction& modifierAction, const ModifierDimension& modifierDimension);
    void addParameter(const KernelId id, const std::string& name, const std::vector<double>& values);
    void addLocalMemoryModifier(const KernelId id, const std::string& parameterName, const ArgumentId argumentId,
        const ModifierAction& modifierAction);
    void addConstraint(const KernelId id, const std::function<bool(std::vector<size_t>)>& constraintFunction,
        const std::vector<std::string>& parameterNames);
    void setArguments(const KernelId id, const std::vector<ArgumentId>& argumentIds);
    void setTuningManipulatorFlag(const KernelId id, const bool flag);
    void addCompositionKernelParameter(const KernelId compositionId, const KernelId kernelId, const std::string& parameterName,
        const std::vector<size_t>& parameterValues, const ModifierType& modifierType, const ModifierAction& modifierAction,
        const ModifierDimension& modifierDimension);
    void addCompositionKernelLocalMemoryModifier(const KernelId compositionId, const KernelId kernelId, const std::string& parameterName,
        const ArgumentId argumentId, const ModifierAction& modifierAction);
    void setCompositionKernelArguments(const KernelId compositionId, const KernelId kernelId, const std::vector<ArgumentId>& argumentIds);

    // Getters
    const Kernel& getKernel(const KernelId id) const;
    Kernel& getKernel(const KernelId id);
    size_t getCompositionCount() const;
    const KernelComposition& getKernelComposition(const KernelId id) const;
    KernelComposition& getKernelComposition(const KernelId id);
    bool isKernel(const KernelId id) const;
    bool isComposition(const KernelId id) const;

private:
    // Attributes
    KernelId nextId;
    std::vector<Kernel> kernels;
    std::vector<KernelComposition> kernelCompositions;
    DeviceInfo currentDeviceInfo;

    // Helper methods
    std::string loadFileToString(const std::string& filePath) const;
    void computeConfigurations(const KernelId kernelId, const size_t currentParameterIndex, const std::vector<KernelParameter>& parameters,
        const std::vector<KernelConstraint>& constraints, const std::vector<ParameterPair>& parameterPairs, const DimensionVector& globalSize,
        const DimensionVector& localSize, const std::vector<LocalMemoryModifier>& modifiers, std::vector<KernelConfiguration>& finalResult) const;
    void computeCompositionConfigurations(const size_t currentParameterIndex, const std::vector<KernelParameter>& parameters,
        const std::vector<KernelConstraint>& constraints, const std::vector<ParameterPair>& parameterPairs,
        const std::vector<std::pair<KernelId, DimensionVector>>& globalSizes, const std::vector<std::pair<KernelId, DimensionVector>>& localSizes,
        const std::vector<std::pair<KernelId, std::vector<LocalMemoryModifier>>>& modifiers, std::vector<KernelConfiguration>& finalResult) const;
    bool configurationIsValid(const KernelConfiguration& configuration, const std::vector<KernelConstraint>& constraints) const;
};

} // namespace ktt
