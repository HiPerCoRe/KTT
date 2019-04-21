#pragma once

#include <map>
#include <vector>
#include <enum/dimension_vector_type.h>
#include <kernel/kernel.h>
#include <kernel/kernel_composition.h>
#include <kernel/kernel_configuration.h>

namespace ktt
{

class KernelManager
{
public:
    // Constructor
    KernelManager();

    // Core methods
    KernelId addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize, const DimensionVector& localSize);
    KernelId addKernelFromFile(const std::string& filePath, const std::string& kernelName, const DimensionVector& globalSize,
        const DimensionVector& localSize);
    KernelId addKernelComposition(const std::string& compositionName, const std::vector<KernelId>& kernelIds);
    std::string getKernelSourceWithDefines(const KernelId id, const KernelConfiguration& configuration) const;
    std::string getKernelSourceWithDefines(const KernelId id, const std::vector<ParameterPair>& configuration) const;
    KernelConfiguration getKernelConfiguration(const KernelId id, const std::vector<ParameterPair>& parameterPairs) const;
    KernelConfiguration getKernelCompositionConfiguration(const KernelId compositionId, const std::vector<ParameterPair>& parameterPairs) const;

    // Kernel modification methods
    void addParameter(const KernelId id, const std::string& name, const std::vector<size_t>& values);
    void addParameter(const KernelId id, const std::string& name, const std::vector<double>& values);
    void addConstraint(const KernelId id, const std::vector<std::string>& parameterNames,
        const std::function<bool(const std::vector<size_t>&)>& constraintFunction);
    void addParameterPack(const KernelId id, const std::string& packName, const std::vector<std::string>& parameterNames);
    void setThreadModifier(const KernelId id, const ModifierType modifierType, const ModifierDimension modifierDimension,
        const std::vector<std::string>& parameterNames, const std::function<size_t(const size_t, const std::vector<size_t>&)>& modifierFunction);
    void setLocalMemoryModifier(const KernelId id, const ArgumentId argumentId, const std::vector<std::string>& parameterNames,
        const std::function<size_t(const size_t, const std::vector<size_t>&)>& modifierFunction);
    void setCompositionKernelThreadModifier(const KernelId compositionId, const KernelId kernelId, const ModifierType modifierType,
        const ModifierDimension modifierDimension, const std::vector<std::string>& parameterNames,
        const std::function<size_t(const size_t, const std::vector<size_t>&)>& modifierFunction);
    void setCompositionKernelLocalMemoryModifier(const KernelId compositionId, const KernelId kernelId, const ArgumentId argumentId,
        const std::vector<std::string>& parameterNames, const std::function<size_t(const size_t, const std::vector<size_t>&)>& modifierFunction);
    void setArguments(const KernelId id, const std::vector<ArgumentId>& argumentIds);
    void setCompositionKernelArguments(const KernelId compositionId, const KernelId kernelId, const std::vector<ArgumentId>& argumentIds);
    void setTuningManipulatorFlag(const KernelId id, const bool flag);
    void setCompositionKernelProfiling(const KernelId compositionId, const KernelId kernelId, const bool flag);

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

    // Helper methods
    static std::string loadFileToString(const std::string& filePath);
};

} // namespace ktt
