#include <Kernel/ThreadModifier.h>
#include <Utility/StlHelpers.h>

namespace ktt
{

ThreadModifier::ThreadModifier() :
    m_Function(nullptr)
{}

ThreadModifier::ThreadModifier(const std::vector<std::string>& parameters, const std::vector<KernelDefinitionId>& definitions,
    std::function<uint64_t(const uint64_t, const std::vector<uint64_t>&)> function) :
    m_Parameters(parameters),
    m_Definitions(definitions),
    m_Function(function)
{}

const std::vector<std::string>& ThreadModifier::GetParameters() const
{
    return m_Parameters;
}

const std::vector<KernelDefinitionId>& ThreadModifier::GetDefinitions() const
{
    return m_Definitions;
}

uint64_t ThreadModifier::GetModifiedSize(const KernelDefinitionId id, const uint64_t originalSize,
    const std::vector<ParameterPair>& pairs) const
{
    if (!ContainsElement(m_Definitions, id))
    {
        return originalSize;
    }

    std::vector<uint64_t> values = ParameterPair::GetParameterValues<uint64_t>(pairs, m_Parameters);
    return m_Function(originalSize, values);
}

} // namespace ktt
