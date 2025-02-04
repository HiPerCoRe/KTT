#include <algorithm>

#include <Api/KttException.h>
#include <Kernel/KernelManager.h>
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
    const DimensionVector& globalSize, const DimensionVector& localSize, const std::vector<std::string>& typeNames)
{
    const std::string templatedName = KernelDefinition::CreateTemplatedName(name, typeNames);

    for (const auto& pair : m_Definitions)
    {
        if (pair.second->GetName() == name && pair.second->GetTemplatedName() == templatedName)
        {
            throw KttException("Kernel definition with name " + name + " already exists");
        }
    }

    const auto id = m_DefinitionIdGenerator.GenerateId();
    m_Definitions[id] = std::make_unique<KernelDefinition>(id, name, source, globalSize, localSize, typeNames);
    return id;
}

KernelDefinitionId KernelManager::AddKernelDefinitionFromFile(const std::string& name, const std::string& filePath,
    const DimensionVector& globalSize, const DimensionVector& localSize, const std::vector<std::string>& typeNames)
{
    const std::string source = LoadFileToString(filePath);
    return AddKernelDefinition(name, source, globalSize, localSize, typeNames);
}

void KernelManager::RemoveKernelDefinition(const KernelDefinitionId id)
{
    const bool definitionUsed = std::any_of(m_Kernels.cbegin(), m_Kernels.cend(), [id](const auto& pair)
    {
        return pair.second->HasDefinition(id);
    });

    if (definitionUsed)
    {
        throw KttException("Kernel definition with id " + std::to_string(id) +
            " cannot be removed because it is still referenced by at least one kernel");
    }

    const size_t erasedCount = m_Definitions.erase(id);

    if (erasedCount == 0)
    {
        throw KttException("Attempting to remove kernel definition with invalid id: " + std::to_string(id));
    }
}

void KernelManager::SetArguments(const KernelDefinitionId id, const std::vector<ArgumentId>& argumentIds)
{
    auto& definition = GetDefinition(id);
    const auto arguments = m_ArgumentManager.GetArguments(argumentIds);
    definition.SetArguments(arguments);
}

KernelId KernelManager::CreateKernel(const std::string& name, const std::vector<KernelDefinitionId>& definitionIds)
{
    for (const auto& pair : m_Kernels)
    {
        if (pair.second->GetName() == name)
        {
            throw KttException("Kernel with name " + name + " already exists");
        }
    }

    const auto id = m_KernelIdGenerator.GenerateId();
    const auto definitions = GetDefinitionsFromIds(definitionIds);
    m_Kernels[id] = std::make_unique<Kernel>(id, name, definitions);
    return id;
}

void KernelManager::RemoveKernel(const KernelId id)
{
    const size_t erasedCount = m_Kernels.erase(id);

    if (erasedCount == 0)
    {
        throw KttException("Attempting to remove kernel with invalid id: " + std::to_string(id));
    }
}

void KernelManager::AddParameter(const KernelId id, const std::string& name, const std::vector<ParameterValue>& values, const std::string& group)
{
    auto& kernel = GetKernel(id);
    kernel.AddParameter(KernelParameter(name, values, group));
}

void KernelManager::AddScriptParameter(const KernelId id, const std::string& name, const ParameterValueType valueType, const std::string& valueScript,
    const std::string& group)
{
    auto& kernel = GetKernel(id);
    kernel.AddParameter(KernelParameter(name, valueType, valueScript, group));
}

void KernelManager::AddConstraint(const KernelId id, const std::vector<std::string>& parameters, ConstraintFunction function)
{
    auto& kernel = GetKernel(id);
    kernel.AddConstraint(parameters, function);
}

void KernelManager::AddGenericConstraint(const KernelId id, const std::vector<std::string>& parameters, GenericConstraintFunction function)
{
    auto& kernel = GetKernel(id);
    kernel.AddGenericConstraint(parameters, function);
}

void KernelManager::AddScriptConstraint(const KernelId id, const std::vector<std::string>& parameters, const std::string& script)
{
    auto& kernel = GetKernel(id);
    kernel.AddScriptConstraint(parameters, script);
}

void KernelManager::AddThreadModifier(const KernelId id, const std::vector<KernelDefinitionId>& definitionIds, const ModifierType type,
    const ModifierDimension dimension, const std::vector<std::string>& parameters,  ModifierFunction function)
{
    auto& kernel = GetKernel(id);
    const ThreadModifier modifier(parameters, definitionIds, function);
    kernel.AddThreadModifier(type, dimension, modifier);
}

void KernelManager::AddScriptThreadModifier(const KernelId id, const std::vector<KernelDefinitionId>& definitionIds, const ModifierType type,
    const ModifierDimension dimension, const std::string& script)
{
    auto& kernel = GetKernel(id);
    const ThreadModifier modifier(definitionIds, script);
    kernel.AddThreadModifier(type, dimension, modifier);
}

void KernelManager::SetProfiledDefinitions(const KernelId id, const std::vector<KernelDefinitionId>& definitionIds)
{
    auto& kernel = GetKernel(id);
    const auto definitions = GetDefinitionsFromIds(definitionIds);
    kernel.SetProfiledDefinitions(definitions);
}

void KernelManager::SetLauncher(const KernelId id, KernelLauncher launcher)
{
    auto& kernel = GetKernel(id);
    kernel.SetLauncher(launcher);
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

KernelDefinitionId KernelManager::GetDefinitionId(const std::string& name, const std::vector<std::string>& typeNames) const
{
    const auto templatedName = KernelDefinition::CreateTemplatedName(name, typeNames);

    const auto iterator = std::find_if(m_Definitions.cbegin(), m_Definitions.cend(), [&name, &templatedName](const auto& pair)
    {
        return pair.second->GetName() == name && pair.second->GetTemplatedName() == templatedName;
    });

    if (iterator == m_Definitions.cend())
    {
        return InvalidKernelDefinitionId;
    }

    return iterator->first;
}

bool KernelManager::IsArgumentUsed(const ArgumentId& id) const
{
    for (const auto& definition : m_Definitions)
    {
        if (definition.second->HasArgument(id))
        {
            return true;
        }
    }

    return false;
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
