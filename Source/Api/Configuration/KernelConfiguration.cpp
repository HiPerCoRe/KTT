#include <Api/Configuration/KernelConfiguration.h>
#include <Utility/ErrorHandling/KttException.h>

namespace ktt
{

KernelConfiguration::KernelConfiguration()
{}

KernelConfiguration::KernelConfiguration(const std::vector<ParameterPair>& pairs) :
    m_Pairs(pairs)
{
    if (pairs.empty())
    {
        throw KttException("Valid kernel configuration must have at least one parameter pair");
    }
}

const std::vector<ParameterPair>& KernelConfiguration::GetPairs() const
{
    return m_Pairs;
}

bool KernelConfiguration::IsValid() const
{
    return !m_Pairs.empty();
}

std::string KernelConfiguration::GeneratePrefix() const
{
    std::string result;

    for (const auto& pair : m_Pairs)
    {
        result += std::string("#define ") + pair.GetString() + std::string("\n");
    }

    return result;
}

} // namespace ktt
