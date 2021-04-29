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
        const DimensionVector& localSize, const std::string& typeName = "");
    KernelDefinitionId AddKernelDefinitionFromFile(const std::string& name, const std::string& filePath,
        const DimensionVector& globalSize, const DimensionVector& localSize, const std::string& typeName = "");
    void RemoveKernelDefinition(const KernelDefinitionId id);
    void SetArguments(const KernelDefinitionId id, const std::vector<ArgumentId>& argumentIds);
    
    KernelId CreateKernel(const std::string& name, const std::vector<KernelDefinitionId>& definitionIds);
    void RemoveKernel(const KernelId id);
    void AddParameter(const KernelId id, const std::string& name, const std::vector<uint64_t>& values, const std::string& group);
    void AddParameter(const KernelId id, const std::string& name, const std::vector<double>& values, const std::string& group);
    void AddConstraint(const KernelId id, const std::vector<std::string>& parameters, ConstraintFunction function);
    void AddThreadModifier(const KernelId id, const std::vector<KernelDefinitionId>& definitionIds, const ModifierType type,
        const ModifierDimension dimension, const std::vector<std::string>& parameters, ModifierFunction function);
    void SetProfiledDefinitions(const KernelId id, const std::vector<KernelDefinitionId>& definitionIds);
    void SetLauncher(const KernelId id, KernelLauncher launcher);

    const Kernel& GetKernel(const KernelId id) const;
    Kernel& GetKernel(const KernelId id);
    const KernelDefinition& GetDefinition(const KernelDefinitionId id) const;
    KernelDefinition& GetDefinition(const KernelDefinitionId id);
    bool IsArgumentUsed(const ArgumentId id) const;

private:
    KernelArgumentManager& m_ArgumentManager;
    IdGenerator<KernelId> m_KernelIdGenerator;
    IdGenerator<KernelDefinitionId> m_DefinitionIdGenerator;
    std::map<KernelId, std::unique_ptr<Kernel>> m_Kernels;
    std::map<KernelDefinitionId, std::unique_ptr<KernelDefinition>> m_Definitions;

    const std::vector<const KernelDefinition*> GetDefinitionsFromIds(const std::vector<KernelDefinitionId>& ids) const;
};

} // namespace ktt
