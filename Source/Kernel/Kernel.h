#pragma once

#include <functional>
#include <map>
#include <queue>
#include <set>
#include <string>
#include <vector>

#include <Api/Configuration/KernelConfiguration.h>
#include <Kernel/KernelDefinition.h>
#include <Kernel/KernelConstraint.h>
#include <Kernel/KernelParameter.h>
#include <Kernel/KernelParameterGroup.h>
#include <Kernel/ModifierDimension.h>
#include <Kernel/ModifierType.h>
#include <Kernel/ThreadModifier.h>
#include <KttTypes.h>

namespace ktt
{

class Kernel
{
public:
    explicit Kernel(const KernelId id, const std::string& name, const std::vector<const KernelDefinition*>& definitions);

    void AddParameter(const KernelParameter& parameter);
    void AddConstraint(const std::vector<std::string>& parameterNames, ConstraintFunction function);
    void AddGenericConstraint(const std::vector<std::string>& parameterNames, GenericConstraintFunction function);
    void AddScriptConstraint(const std::vector<std::string>& parameterNames, const std::string& script);
    void AddThreadModifier(const ModifierType type, const ModifierDimension dimension, const ThreadModifier& modifier);
    void SetProfiledDefinitions(const std::vector<const KernelDefinition*>& definitions);
    void SetLauncher(KernelLauncher launcher);

    KernelId GetId() const;
    const std::string& GetName() const;
    const KernelDefinition& GetPrimaryDefinition() const;
    const KernelDefinition& GetDefinition(const KernelDefinitionId id) const;
    const std::vector<const KernelDefinition*>& GetDefinitions() const;
    const std::vector<const KernelDefinition*>& GetProfiledDefinitions() const;
    const std::set<KernelParameter>& GetParameters() const;
    const std::vector<KernelConstraint>& GetConstraints() const;
    std::vector<KernelArgument*> GetVectorArguments() const;
    KernelLauncher GetLauncher() const;

    bool HasLauncher() const;
    bool HasDefinition(const KernelDefinitionId id) const;
    bool HasParameter(const std::string& name) const;
    bool IsComposite() const;

    KernelConfiguration CreateConfiguration(const ParameterInput& parameters) const;
    std::vector<KernelParameterGroup> GenerateParameterGroups() const;
    void EnumerateNeighbourConfigurations(const KernelConfiguration& configuration,
        std::function<bool(const KernelConfiguration&, const uint64_t)> enumerator) const;

    DimensionVector GetModifiedGlobalSize(const KernelDefinitionId id, const std::vector<ParameterPair>& pairs) const;
    DimensionVector GetModifiedLocalSize(const KernelDefinitionId id, const std::vector<ParameterPair>& pairs) const;

private:
    KernelId m_Id;
    std::string m_Name;
    std::vector<const KernelDefinition*> m_Definitions;
    std::vector<const KernelDefinition*> m_ProfiledDefinitions;
    std::set<KernelParameter> m_Parameters;
    std::vector<KernelConstraint> m_Constraints;
    std::map<ModifierType, std::map<ModifierDimension, std::vector<ThreadModifier>>> m_Modifiers;
    KernelLauncher m_Launcher;

    std::vector<const KernelParameter*> PreprocessConstraintParameters(const std::vector<std::string>& parameterNames,
        const bool genericConstraint) const;
    std::vector<const KernelConstraint*> GetConstraintsForParameters(const std::vector<const KernelParameter*>& parameters) const;
    const KernelParameter& GetParamater(const std::string& name) const;
    DimensionVector GetModifiedSize(const KernelDefinitionId id, const ModifierType type,
        const std::vector<ParameterPair>& pairs) const;
    void EnumerateNeighbours(const KernelConfiguration& configuration, const KernelParameter* neighbourParameter,
        const std::set<const KernelParameter*>& enumeratedParameters, std::set<std::set<const KernelParameter*>>& enumeratedSets,
        std::function<bool(const KernelConfiguration&, const uint64_t)> enumerator,
        std::queue<std::tuple<const KernelConfiguration&, const KernelParameter*, const std::set<const KernelParameter*>&>>& queue) const;
};

} // namespace ktt
