#pragma once

#include <functional>
#include <map>
#include <string>
#include <vector>

#include <Kernel/KernelDefinition.h>
#include <Kernel/KernelConstraint.h>
#include <Kernel/KernelParameter.h>
#include <Kernel/ModifierDimension.h>
#include <Kernel/ModifierType.h>
#include <Kernel/ThreadModifier.h>
#include <KttTypes.h>

namespace ktt
{

class Kernel
{
public:
    explicit Kernel(const KernelId id, const std::vector<const KernelDefinition*>& definitions);

    void AddParameter(const KernelParameter& parameter);
    void AddConstraint(const KernelConstraint& constraint);
    void SetThreadModifier(const ModifierType type, const ModifierDimension dimension, const ThreadModifier& modifier);
    void SetProfiledDefinitions(const std::vector<const KernelDefinition*>& definitions);

    KernelId GetId() const;
    const KernelDefinition& GetPrimaryDefinition() const;
    const KernelDefinition& GetDefinition(const KernelDefinitionId id) const;
    const std::vector<const KernelDefinition*>& GetDefinitions() const;
    const std::vector<const KernelDefinition*>& GetProfiledDefinitions() const;
    const std::vector<KernelParameter>& GetParameters() const;
    const std::vector<KernelConstraint>& GetConstraints() const;

    bool HasDefinition(const KernelDefinitionId id) const;
    bool HasParameter(const std::string& parameter) const;
    bool IsComposite() const;

    uint64_t GetConfigurationsCount() const;
    std::vector<ParameterPair> GetPairsForIndex(const uint64_t index) const;
    uint64_t GetIndexForPairs(const std::vector<ParameterPair>& pairs) const;

    DimensionVector GetModifiedGlobalSize(const KernelDefinitionId id, const std::vector<ParameterPair>& pairs) const;
    DimensionVector GetModifiedLocalSize(const KernelDefinitionId id, const std::vector<ParameterPair>& pairs) const;

private:
    KernelId m_Id;
    std::vector<const KernelDefinition*> m_Definitions;
    std::vector<const KernelDefinition*> m_ProfiledDefinitions;
    std::vector<KernelParameter> m_Parameters;
    std::vector<KernelConstraint> m_Constraints;
    std::map<ModifierType, std::map<ModifierDimension, ThreadModifier>> m_Modifiers;

    DimensionVector GetModifiedSize(const KernelDefinitionId id, const ModifierType type,
        const std::vector<ParameterPair>& pairs) const;
};

} // namespace ktt
