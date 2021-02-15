#include <algorithm>

#include <Kernel/Kernel.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/ErrorHandling/KttException.h>
#include <Utility/StlHelpers.h>

namespace ktt
{

Kernel::Kernel(const KernelId id, const std::vector<const KernelDefinition*>& definitions) :
    m_Id(id),
    m_Definitions(definitions)
{
    KttAssert(!m_Definitions.empty(), "Each kernel must have at least one definition");
    m_ProfiledDefinitions.push_back(&GetPrimaryDefinition());
}

void Kernel::AddParameter(const KernelParameter& parameter)
{
    if (ContainsKey(m_Parameters, parameter))
    {
        throw KttException("Kernel parameter with name " + parameter.GetName() + " already exists");
    }

    m_Parameters.insert(parameter);
}

void Kernel::AddConstraint(const KernelConstraint& constraint)
{
    for (const auto& name : constraint.GetParameters())
    {
        const auto& parameter = GetParamater(name);

        if (parameter.HasValuesDouble())
        {
            throw KttException("Kernel parameter with name " + name
                + " has floating-point values and cannot be used by kernel constraints");
        }
    }

    m_Constraints.push_back(constraint);
}

void Kernel::SetThreadModifier(const ModifierType type, const ModifierDimension dimension, const ThreadModifier& modifier)
{
    for (const auto& name : modifier.GetParameters())
    {
        const auto& parameter = GetParamater(name);

        if (parameter.HasValuesDouble())
        {
            throw KttException("Kernel parameter with name " + name
                + " has floating-point values and cannot be used by thread modifiers");
        }
    }

    for (const auto id : modifier.GetDefinitions())
    {
        if (!HasDefinition(id))
        {
            throw KttException("Kernel with id " + std::to_string(m_Id) + " does not contain definition with id "
                + std::to_string(id) + " specified by thread modifier");
        }
    }

    auto& specificModifiers = m_Modifiers[type];
    specificModifiers[dimension] = modifier;
}

void Kernel::SetProfiledDefinitions(const std::vector<const KernelDefinition*>& definitions)
{
    for (const auto* definition : definitions)
    {
        if (!HasDefinition(definition->GetId()))
        {
            throw KttException("Kernel with id " + std::to_string(m_Id) + " does not contain definition with id "
                + std::to_string(definition->GetId()));
        }
    }

    m_ProfiledDefinitions = definitions;
}

void Kernel::SetLauncher(KernelLauncher launcher)
{
    m_Launcher = launcher;
}

KernelId Kernel::GetId() const
{
    return m_Id;
}

const KernelDefinition& Kernel::GetPrimaryDefinition() const
{
    KttAssert(!IsComposite(), "Only simple kernels have primary definition");
    return *m_Definitions[0];
}

const KernelDefinition& Kernel::GetDefinition(const KernelDefinitionId id) const
{
    KttAssert(HasDefinition(id), "Invalid kernel definition");

    for (const auto* definition : m_Definitions)
    {
        if (definition->GetId() == id)
        {
            return *definition;
        }
    }

    return GetPrimaryDefinition();
}

const std::vector<const KernelDefinition*>& Kernel::GetDefinitions() const
{
    return m_Definitions;
}

const std::vector<const KernelDefinition*>& Kernel::GetProfiledDefinitions() const
{
    return m_ProfiledDefinitions;
}

const std::set<KernelParameter>& Kernel::GetParameters() const
{
    return m_Parameters;
}

const std::vector<KernelConstraint>& Kernel::GetConstraints() const
{
    return m_Constraints;
}

std::vector<KernelArgument*> Kernel::GetVectorArguments() const
{
    std::set<KernelArgument*> arguments;

    for (const auto* definition : m_Definitions)
    {
        for (auto* argument : definition->GetVectorArguments())
        {
            arguments.insert(argument);
        }
    }

    std::vector<KernelArgument*> result(arguments.cbegin(), arguments.cend());
    return result;
}

KernelLauncher Kernel::GetLauncher() const
{
    return m_Launcher;
}

bool Kernel::HasLauncher() const
{
    return static_cast<bool>(m_Launcher);
}

bool Kernel::HasDefinition(const KernelDefinitionId id) const
{
    return ContainsElementIf(m_Definitions, [id](const auto* definition)
    {
        return definition->GetId() == id;
    });
}

bool Kernel::HasParameter(const std::string& name) const
{
    for (const auto& parameter : m_Parameters)
    {
        if (parameter.GetName() == name)
        {
            return true;
        }
    }

    return false;
}

bool Kernel::IsComposite() const
{
    return m_Definitions.size() > 1;
}

bool Kernel::HasWritableZeroCopyArgument() const
{
    for (const auto* argument : GetVectorArguments())
    {
        if (argument->GetMemoryLocation() == ArgumentMemoryLocation::HostZeroCopy
            && argument->GetAccessType() != ArgumentAccessType::ReadOnly)
        {
            return true;
        }
    }

    return false;
}

KernelConfiguration Kernel::CreateConfiguration(const ParameterInput& parameters) const
{
    std::vector<ParameterPair> pairs;

    for (const auto& pair : parameters)
    {
        const auto& parameter = GetParamater(pair.first);

        if (parameter.HasValuesDouble())
        {
            if (!std::holds_alternative<double>(pair.second))
            {
                throw KttException("Value type mismatch for parameter with name " + pair.first);
            }

            pairs.emplace_back(parameter.GetName(), std::get<double>(pair.second));
        }
        else
        {
            if (!std::holds_alternative<uint64_t>(pair.second))
            {
                throw KttException("Value type mismatch for parameter with name " + pair.first);
            }

            pairs.emplace_back(parameter.GetName(), std::get<uint64_t>(pair.second));
        }
    }

    for (const auto& parameter : m_Parameters)
    {
        const bool parameterIncluded = ContainsElementIf(pairs, [&parameter](const auto& pair)
        {
            return pair.GetName() == parameter.GetName();
        });

        if (!parameterIncluded)
        {
            pairs.push_back(parameter.GeneratePair(0));
        }
    }


    return KernelConfiguration(pairs);
}

std::vector<KernelParameterGroup> Kernel::GenerateParameterGroups() const
{
    std::map<std::string, std::vector<const KernelParameter*>> groupedParameters;

    for (const auto& parameter : m_Parameters)
    {
        groupedParameters[parameter.GetGroup()].push_back(&parameter);
    }

    std::vector<KernelParameterGroup> result;

    for (const auto& groupPair : groupedParameters)
    {
        result.emplace_back(groupPair.first, groupPair.second);
    }

    std::sort(result.begin(), result.end(), [](const auto& first, const auto& second)
    {
        return first.GetConfigurationsCount() < second.GetConfigurationsCount();
    });

    return result;
}

uint64_t Kernel::GetConfigurationsCount() const
{
    uint64_t result = 1;

    for (const auto& parameter : m_Parameters)
    {
        result *= static_cast<uint64_t>(parameter.GetValuesCount());
    }

    return result;
}

std::vector<ParameterPair> Kernel::GetPairsForIndex(const uint64_t index) const
{
    std::vector<ParameterPair> result;
    uint64_t currentIndex = index;

    for (const auto& parameter : m_Parameters)
    {
        const size_t valuesCount = parameter.GetValuesCount();
        const size_t parameterIndex = currentIndex % valuesCount;

        result.push_back(parameter.GeneratePair(parameterIndex));
        currentIndex /= valuesCount;
    }

    return result;
}

uint64_t Kernel::GetIndexForPairs(const std::vector<ParameterPair>& pairs) const
{
    for (const auto& pair : pairs)
    {
        if (!HasParameter(pair.GetName()))
        {
            throw KttException("Kernel parameter with name " + pair.GetName() + " does not exist");
        }
    }

    for (const auto& parameter : m_Parameters)
    {
        const bool hasPair = ContainsElementIf(pairs, [&parameter](const auto& pair)
        {
            return pair.GetName() == parameter.GetName();
        });

        if (!hasPair)
        {
            throw KttException("Kernel parameter with name " + parameter.GetName() + " is not present in pairs");
        }
    }

    uint64_t result = 0;
    uint64_t multiplier = 1;

    for (const auto& parameter : m_Parameters)
    {
        const ParameterPair* currentPair = nullptr;

        for (const auto& pair : pairs)
        {
            if (parameter.GetName() == pair.GetName())
            {
                currentPair = &pair;
                break;
            }
        }

        const auto parameterPairs = parameter.GeneratePairs();

        for (size_t i = 0; i < parameterPairs.size(); ++i)
        {
            if (currentPair->HasSameValue(parameterPairs[i]))
            {
                result += multiplier * i;
                break;
            }
        }

        multiplier *= parameterPairs.size();
    }

    return result;
}

DimensionVector Kernel::GetModifiedGlobalSize(const KernelDefinitionId id, const std::vector<ParameterPair>& pairs) const
{
    return GetModifiedSize(id, ModifierType::Global, pairs);
}

DimensionVector Kernel::GetModifiedLocalSize(const KernelDefinitionId id, const std::vector<ParameterPair>& pairs) const
{
    return GetModifiedSize(id, ModifierType::Local, pairs);
}

const KernelParameter& Kernel::GetParamater(const std::string& name) const
{
    for (const auto& parameter : m_Parameters)
    {
        if (parameter.GetName() == name)
        {
            return parameter;
        }
    }

    throw KttException("Kernel parameter with name " + name + " does not exist");
}

DimensionVector Kernel::GetModifiedSize(const KernelDefinitionId id, const ModifierType type,
    const std::vector<ParameterPair>& pairs) const
{
    KttAssert(HasDefinition(id), "Invalid definition id");
    const auto& definition = GetDefinition(id);
    const auto& defaultSize = type == ModifierType::Global ? definition.GetGlobalSize() : definition.GetLocalSize();

    if (!ContainsKey(m_Modifiers, type))
    {
        return defaultSize;
    }

    const auto& modifiersPair = *m_Modifiers.find(type);
    const auto& specificModifiers = modifiersPair.second;
    DimensionVector result;

    for (int i = 0; i <= static_cast<int>(ModifierDimension::Z); ++i)
    {
        const auto dimension = static_cast<ModifierDimension>(i);
        size_t dimensionSize = defaultSize.GetSize(dimension);

        if (ContainsKey(specificModifiers, dimension))
        {
            const auto& pair = *specificModifiers.find(dimension);
            dimensionSize = pair.second.GetModifiedSize(id, dimensionSize, pairs);
        }

        result.SetSize(dimension, dimensionSize);
    }

    return result;
}

} // namespace ktt
