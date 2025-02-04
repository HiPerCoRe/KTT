#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <string>

#include <Kernel/Kernel.h>
#include <Kernel/KernelDefinition.h>
#include <KernelArgument/KernelArgumentManager.h>
#include <Utility/IdGenerator.h>
#include <KttTypes.h>

namespace ktt
{

class KernelManager
{
public:
    KernelManager(KernelArgumentManager& argumentManager);

    KernelDefinitionId AddKernelDefinition(const std::string& name, const std::string& source, const DimensionVector& globalSize,
        const DimensionVector& localSize, const std::vector<std::string>& typeNames = {});
    KernelDefinitionId AddKernelDefinitionFromFile(const std::string& name, const std::string& filePath,
        const DimensionVector& globalSize, const DimensionVector& localSize, const std::vector<std::string>& typeNames = {});
    void RemoveKernelDefinition(const KernelDefinitionId id);
    void SetArguments(const KernelDefinitionId id, const std::vector<ArgumentId>& argumentIds);
    
    KernelId CreateKernel(const std::string& name, const std::vector<KernelDefinitionId>& definitionIds);
    void RemoveKernel(const KernelId id);
    void AddParameter(const KernelId id, const std::string& name, const std::vector<ParameterValue>& values, const std::string& group);
    void AddScriptParameter(const KernelId id, const std::string& name, const ParameterValueType valueType, const std::string& valueScript,
        const std::string& group);
    void AddConstraint(const KernelId id, const std::vector<std::string>& parameters, ConstraintFunction function);
    void AddGenericConstraint(const KernelId id, const std::vector<std::string>& parameters, GenericConstraintFunction function);
    void AddScriptConstraint(const KernelId id, const std::vector<std::string>& parameters, const std::string& script);
    void AddThreadModifier(const KernelId id, const std::vector<KernelDefinitionId>& definitionIds, const ModifierType type,
        const ModifierDimension dimension, const std::vector<std::string>& parameters, ModifierFunction function);
    void AddScriptThreadModifier(const KernelId id, const std::vector<KernelDefinitionId>& definitionIds, const ModifierType type,
        const ModifierDimension dimension, const std::string& script);
    void SetProfiledDefinitions(const KernelId id, const std::vector<KernelDefinitionId>& definitionIds);
    void SetLauncher(const KernelId id, KernelLauncher launcher);

    const Kernel& GetKernel(const KernelId id) const;
    Kernel& GetKernel(const KernelId id);
    const KernelDefinition& GetDefinition(const KernelDefinitionId id) const;
    KernelDefinition& GetDefinition(const KernelDefinitionId id);
    KernelDefinitionId GetDefinitionId(const std::string& name, const std::vector<std::string>& typeNames = {}) const;
    bool IsArgumentUsed(const ArgumentId& id) const;

private:
    KernelArgumentManager& m_ArgumentManager;
    IdGenerator<KernelId> m_KernelIdGenerator;
    IdGenerator<KernelDefinitionId> m_DefinitionIdGenerator;
    std::map<KernelId, std::unique_ptr<Kernel>> m_Kernels;
    std::map<KernelDefinitionId, std::unique_ptr<KernelDefinition>> m_Definitions;

    const std::vector<const KernelDefinition*> GetDefinitionsFromIds(const std::vector<KernelDefinitionId>& ids) const;
};

} // namespace ktt
