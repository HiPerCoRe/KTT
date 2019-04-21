#pragma once

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <api/dimension_vector.h>
#include <api/parameter_pair.h>
#include <dto/local_memory_modifier.h>
#include <enum/modifier_dimension.h>
#include <enum/modifier_type.h>
#include <kernel/kernel_constraint.h>
#include <kernel/kernel_parameter.h>
#include <kernel/kernel_parameter_pack.h>
#include <ktt_types.h>

namespace ktt
{

class Kernel
{
public:
    // Constructor
    explicit Kernel(const KernelId id, const std::string& source, const std::string& name, const DimensionVector& globalSize,
        const DimensionVector& localSize);

    // Core methods
    void addParameter(const KernelParameter& parameter);
    void addConstraint(const KernelConstraint& constraint);
    void addParameterPack(const KernelParameterPack& pack);
    void setThreadModifier(const ModifierType modifierType, const ModifierDimension modifierDimension,
        const std::vector<std::string>& parameterNames, const std::function<size_t(const size_t, const std::vector<size_t>&)>& modifierFunction);
    void setLocalMemoryModifier(const ArgumentId argumentId, const std::vector<std::string>& parameterNames,
        const std::function<size_t(const size_t, const std::vector<size_t>&)>& modifierFunction);
    void setArguments(const std::vector<ArgumentId>& argumentIds);
    void setTuningManipulatorFlag(const bool flag);

    // Getters
    KernelId getId() const;
    const std::string& getSource() const;
    const std::string& getName() const;
    const DimensionVector& getGlobalSize() const;
    DimensionVector getModifiedGlobalSize(const std::vector<ParameterPair>& parameterPairs) const;
    const DimensionVector& getLocalSize() const;
    DimensionVector getModifiedLocalSize(const std::vector<ParameterPair>& parameterPairs) const;
    const std::vector<KernelParameter>& getParameters() const;
    const std::vector<KernelConstraint>& getConstraints() const;
    const std::vector<KernelParameterPack>& getParameterPacks() const;
    std::vector<KernelParameter> getParametersOutsidePacks() const;
    std::vector<KernelParameter> getParametersForPack(const std::string& pack) const;
    std::vector<KernelParameter> getParametersForPack(const KernelParameterPack& pack) const;
    size_t getArgumentCount() const;
    const std::vector<ArgumentId>& getArgumentIds() const;
    std::vector<LocalMemoryModifier> getLocalMemoryModifiers(const std::vector<ParameterPair>& parameterPairs) const;
    bool hasParameter(const std::string& parameterName) const;
    bool hasTuningManipulator() const;

private:
    // Attributes
    KernelId id;
    std::string source;
    std::string name;
    DimensionVector globalSize;
    DimensionVector localSize;
    std::vector<KernelParameter> parameters;
    std::vector<KernelConstraint> constraints;
    std::vector<KernelParameterPack> parameterPacks;
    std::vector<ArgumentId> argumentIds;
    std::array<std::vector<std::string>, 3> globalThreadModifierNames;
    std::array<std::function<size_t(const size_t, const std::vector<size_t>&)>, 3> globalThreadModifiers;
    std::array<std::vector<std::string>, 3> localThreadModifierNames;
    std::array<std::function<size_t(const size_t, const std::vector<size_t>&)>, 3> localThreadModifiers;
    std::map<ArgumentId, std::vector<std::string>> localMemoryModifierNames;
    std::map<ArgumentId, std::function<size_t(const size_t, const std::vector<size_t>&)>> localMemoryModifiers;
    bool tuningManipulatorFlag;

    void validateModifierParameters(const std::vector<std::string>& parameterNames) const;
};

} // namespace ktt
