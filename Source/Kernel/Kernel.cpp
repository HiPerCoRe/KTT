#include <algorithm>

#include <Api/KttException.h>
#include <Kernel/Kernel.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/StlHelpers.h>

namespace ktt
{

Kernel::Kernel(const KernelId id, const std::string& name, const std::vector<const KernelDefinition*>& definitions) :
    m_Id(id),
    m_Name(name),
    m_Definitions(definitions)
{
    KttAssert(!m_Definitions.empty(), "Each kernel must have at least one definition");
    m_ProfiledDefinitions.push_back(definitions[0]);
}

void Kernel::AddParameter(const KernelParameter& parameter)
{
    if (ContainsKey(m_Parameters, parameter))
    {
        throw KttException("Kernel parameter with name " + parameter.GetName() + " already exists");
    }

    m_Parameters.insert(parameter);
}

void Kernel::AddConstraint(const std::vector<std::string>& parameterNames, ConstraintFunction function)
{
    const std::vector<const KernelParameter*> parameters = PreprocessConstraintParameters(parameterNames, false);
    m_Constraints.emplace_back(parameters, function);
}

void Kernel::AddGenericConstraint(const std::vector<std::string>& parameterNames, GenericConstraintFunction function)
{
    const std::vector<const KernelParameter*> parameters = PreprocessConstraintParameters(parameterNames, true);
    m_Constraints.emplace_back(parameters, function);
}

void Kernel::AddScriptConstraint(const std::vector<std::string>& parameterNames, const std::string& script)
{
    const std::vector<const KernelParameter*> parameters = PreprocessConstraintParameters(parameterNames, true);
    m_Constraints.emplace_back(parameters, script);
}

void Kernel::AddThreadModifier(const ModifierType type, const ModifierDimension dimension, const ThreadModifier& modifier)
{
    for (const auto& name : modifier.GetParameters())
    {
        const auto& parameter = GetParamater(name);

        if (parameter.GetValueType() != ParameterValueType::UnsignedInt)
        {
            throw KttException("Kernel parameter with name " + name
                + " does not have unsigned integer values and cannot be used by thread modifiers");
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
    specificModifiers[dimension].push_back(modifier);
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

const std::string& Kernel::GetName() const
{
    return m_Name;
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

KernelConfiguration Kernel::CreateConfiguration(const ParameterInput& parameters) const
{
    std::vector<ParameterPair> pairs;

    for (const auto& pair : parameters)
    {
        const auto& parameter = GetParamater(pair.first);

        if (parameter.GetValueType() != ParameterPair::GetTypeFromValue(pair.second))
        {
            throw KttException("Value type mismatch for parameter with name " + pair.first);
        }

        pairs.emplace_back(parameter.GetName(), pair.second);
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
        const auto constraints = GetConstraintsForParameters(groupPair.second);
        result.emplace_back(groupPair.first, groupPair.second, constraints);
    }

    return result;
}

void Kernel::EnumerateNeighbourConfigurations(const KernelConfiguration& configuration,
    std::function<bool(const KernelConfiguration&, const uint64_t)> enumerator) const
{
    std::set<std::set<const KernelParameter*>> initialSets;
    std::queue<std::tuple<const KernelConfiguration&, const KernelParameter*, const std::set<const KernelParameter*>&>> initialQueue;
    EnumerateNeighbours(configuration, nullptr, std::set<const KernelParameter*>{}, initialSets, enumerator, initialQueue);
}

DimensionVector Kernel::GetModifiedGlobalSize(const KernelDefinitionId id, const std::vector<ParameterPair>& pairs) const
{
    return GetModifiedSize(id, ModifierType::Global, pairs);
}

DimensionVector Kernel::GetModifiedLocalSize(const KernelDefinitionId id, const std::vector<ParameterPair>& pairs) const
{
    return GetModifiedSize(id, ModifierType::Local, pairs);
}

std::vector<const KernelParameter*> Kernel::PreprocessConstraintParameters(const std::vector<std::string>& parameterNames,
    const bool genericConstraint) const
{
    std::vector<const KernelParameter*> parameters;
    std::set<std::string> usedGroups;

    for (const auto& name : parameterNames)
    {
        const auto& parameter = GetParamater(name);

        if (!genericConstraint && parameter.GetValueType() != ParameterValueType::UnsignedInt)
        {
            throw KttException("Kernel parameter with name " + name
                + " does not have unsigned integer values and cannot be used by non-generic kernel constraints");
        }

        usedGroups.insert(parameter.GetGroup());

        if (usedGroups.size() > 1)
        {
            throw KttException("Constraint can only be added between parameters that belong to the same group");
        }

        parameters.push_back(&parameter);
    }

    return parameters;
}

std::vector<const KernelConstraint*> Kernel::GetConstraintsForParameters(const std::vector<const KernelParameter*>& parameters) const
{
    std::vector<const KernelConstraint*> result;

    for (const auto& constraint : GetConstraints())
    {
        for (const auto* parameter : parameters)
        {
            if (constraint.AffectsParameter(parameter->GetName()))
            {
                result.push_back(&constraint);
                break;
            }
        }
    }

    return result;
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

            for (const auto& modifier : pair.second)
            {
                dimensionSize = static_cast<size_t>(modifier.GetModifiedSize(id, static_cast<uint64_t>(dimensionSize), pairs));
            }
        }

        result.SetSize(dimension, dimensionSize);
    }

    return result;
}

void Kernel::EnumerateNeighbours(const KernelConfiguration& configuration, const KernelParameter* neighbourParameter,
    const std::set<const KernelParameter*>& enumeratedParameters, std::set<std::set<const KernelParameter*>>& enumeratedSets,
    std::function<bool(const KernelConfiguration&, const uint64_t)> enumerator,
    std::queue<std::tuple<const KernelConfiguration&, const KernelParameter*, const std::set<const KernelParameter*>&>>& queue) const
{
    std::vector<KernelConfiguration> newNeighbours;

    if (neighbourParameter != nullptr)
    {
        newNeighbours = configuration.GenerateNeighbours(neighbourParameter->GetName(), neighbourParameter->GeneratePairs());
    }

    for (const auto& neighbour : newNeighbours)
    {
        const bool interrupt = !enumerator(neighbour, enumeratedParameters.size());

        if (interrupt)
        {
            while (!queue.empty())
            {
                queue.pop();
            }

            return;
        }
    }

    for (const auto& parameter : m_Parameters)
    {
        auto newEnumerated = enumeratedParameters;
        newEnumerated.insert(&parameter);

        if (ContainsKey(enumeratedSets, newEnumerated))
        {
            continue;
        }

        auto iterator = enumeratedSets.insert(newEnumerated);

        if (neighbourParameter == nullptr)
        {
            queue.push({configuration, &parameter, *iterator.first});
            continue;
        }

        for (const auto& neighbour : newNeighbours)
        {
            queue.push({neighbour, &parameter, *iterator.first});
        }
    }

    while (!queue.empty())
    {
        auto next = queue.front();
        queue.pop();
        EnumerateNeighbours(std::get<0>(next), std::get<1>(next), std::get<2>(next), enumeratedSets, enumerator, queue);
    }
}

} // namespace ktt
