#include <Kernel/KernelManager.h>
#include <Utility/ErrorHandling/KttException.h>
#include <Utility/FileSystem.h>
#include <Utility/StlHelpers.h>

namespace ktt
{

KernelManager::KernelManager(KernelArgumentManager& argumentManager) :
    m_ArgumentManager(argumentManager),
    m_KernelIdGenerator(0),
    m_DefinitionIdGenerator(0)
{}

KernelDefinitionId KernelManager::AddKernelDefinition(const std::string& name, const std::string& source,
    const DimensionVector& globalSize, const DimensionVector& localSize)
{
    const auto id = m_DefinitionIdGenerator.GenerateId();
    m_Definitions[id] = std::make_unique<KernelDefinition>(id, name, source, globalSize, localSize);
    return id;
}

KernelDefinitionId KernelManager::AddKernelDefinitionFromFile(const std::string& name, const std::string& filePath,
    const DimensionVector& globalSize, const DimensionVector& localSize)
{
    const std::string source = LoadFileToString(filePath);
    return AddKernelDefinition(name, source, globalSize, localSize);
}

void KernelManager::SetArguments(const KernelDefinitionId id, const std::vector<ArgumentId>& argumentIds)
{
    auto& definition = GetDefinition(id);
    const auto arguments = m_ArgumentManager.GetArguments(argumentIds);
    definition.SetArguments(arguments);
}

KernelId KernelManager::CreateKernel(const std::vector<KernelDefinitionId>& definitionIds)
{
    const auto id = m_KernelIdGenerator.GenerateId();
    const auto definitions = GetDefinitionsFromIds(definitionIds);
    m_Kernels[id] = std::make_unique<Kernel>(id, definitions);
    return id;
}

void KernelManager::AddParameter(const KernelId id, const std::string& name, const std::vector<uint64_t>& values, const std::string& group)
{
    auto& kernel = GetKernel(id);
    kernel.AddParameter(KernelParameter(name, values, group));
}

void KernelManager::AddParameter(const KernelId id, const std::string& name, const std::vector<double>& values, const std::string& group)
{
    auto& kernel = GetKernel(id);
    kernel.AddParameter(KernelParameter(name, values, group));
}

void KernelManager::AddConstraint(const KernelId id, const std::vector<std::string>& parameters,
    std::function<bool(const std::vector<size_t>&)> function)
{
    auto& kernel = GetKernel(id);
    kernel.AddConstraint(KernelConstraint(parameters, function));
}

void KernelManager::SetThreadModifier(const KernelId id, const ModifierType type, const ModifierDimension dimension,
    const std::vector<std::string>& parameters, const std::vector<KernelDefinitionId>& definitionIds,
    std::function<uint64_t(const uint64_t, const std::vector<uint64_t>&)> function)
{
    auto& kernel = GetKernel(id);
    const ThreadModifier modifier(parameters, definitionIds, function);
    kernel.SetThreadModifier(type, dimension, modifier);
}

void KernelManager::SetProfiledDefinitions(const KernelId id, const std::vector<KernelDefinitionId>& definitionIds)
{
    auto& kernel = GetKernel(id);
    const auto definitions = GetDefinitionsFromIds(definitionIds);
    kernel.SetProfiledDefinitions(definitions);
}

const Kernel& KernelManager::GetKernel(const KernelId id) const
{
    if (!ContainsKey(m_Kernels, id))
    {
        throw KttException("Attempting to retrieve kernel with invalid id: " + std::to_string(id));
    }

    return *m_Kernels.find(id)->second;
}

Kernel& KernelManager::GetKernel(const KernelId id)
{
    return const_cast<Kernel&>(static_cast<const KernelManager*>(this)->GetKernel(id));
}

const KernelDefinition& KernelManager::GetDefinition(const KernelDefinitionId id) const
{
    if (!ContainsKey(m_Definitions, id))
    {
        throw KttException("Attempting to retrieve kernel definition with invalid id: " + std::to_string(id));
    }

    return *m_Definitions.find(id)->second;
}

KernelDefinition& KernelManager::GetDefinition(const KernelDefinitionId id)
{
    return const_cast<KernelDefinition&>(static_cast<const KernelManager*>(this)->GetDefinition(id));
}

const std::vector<const KernelDefinition*> KernelManager::GetDefinitionsFromIds(const std::vector<KernelDefinitionId>& ids) const
{
    std::vector<const KernelDefinition*> definitions;

    for (const auto id : ids)
    {
        const auto& definition = GetDefinition(id);
        definitions.push_back(&definition);
    }

    return definitions;
}

} // namespace ktt
