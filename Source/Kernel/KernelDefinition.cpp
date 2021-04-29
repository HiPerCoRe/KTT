#include <Api/KttException.h>
#include <Kernel/KernelDefinition.h>
#include <Utility/StlHelpers.h>

namespace ktt
{

KernelDefinition::KernelDefinition(const KernelDefinitionId id, const std::string& name, const std::string& source,
    const DimensionVector& globalSize, const DimensionVector& localSize, const std::string& typeName) :
    m_Id(id),
    m_Name(name),
    m_Source(source),
    m_TypeName(typeName),
    m_GlobalSize(globalSize),
    m_LocalSize(localSize)
{}

void KernelDefinition::SetArguments(const std::vector<KernelArgument*> arguments)
{
    if (!ContainsUniqueElements(arguments))
    {
        throw KttException("Kernel arguments for a single kernel definition must be unique");
    }

    m_Arguments = arguments;
}

KernelDefinitionId KernelDefinition::GetId() const
{
    return m_Id;
}

const std::string& KernelDefinition::GetName() const
{
    return m_Name;
}

const std::string& KernelDefinition::GetSource() const
{
    return m_Source;
}

const std::string& KernelDefinition::GetTypeName() const
{
    return m_TypeName;
}

const DimensionVector& KernelDefinition::GetGlobalSize() const
{
    return m_GlobalSize;
}

const DimensionVector& KernelDefinition::GetLocalSize() const
{
    return m_LocalSize;
}

const std::vector<KernelArgument*>& KernelDefinition::GetArguments() const
{
    return m_Arguments;
}

std::vector<KernelArgument*> KernelDefinition::GetVectorArguments() const
{
    std::vector<KernelArgument*> result;

    for (auto* argument : m_Arguments)
    {
        if (argument->GetMemoryType() == ArgumentMemoryType::Vector)
        {
            result.push_back(argument);
        }
    }

    return result;
}

bool KernelDefinition::HasArgument(const ArgumentId id) const
{
    return ContainsElementIf(m_Arguments, [id](const auto* argument)
    {
        return argument->GetId() == id;
    });
}

} // namespace ktt
