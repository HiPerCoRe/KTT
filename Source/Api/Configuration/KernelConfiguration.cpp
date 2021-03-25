#include <Api/Configuration/KernelConfiguration.h>
#include <Api/KttException.h>
#include <Utility/StlHelpers.h>

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

std::string KernelConfiguration::GetString() const
{
    if (m_Pairs.empty())
    {
        return "empty";
    }

    std::string result;

    for (size_t i = 0; i < m_Pairs.size(); ++i)
    {
        result += m_Pairs[i].GetString();

        if (i + 1 != m_Pairs.size())
        {
            result += ", ";
        }
    }

    return result;
}

void KernelConfiguration::Merge(const KernelConfiguration& other)
{
    for (const auto& otherPair : other.GetPairs())
    {
        const bool includePair = !ContainsElementIf(m_Pairs, [&otherPair](const auto& pair)
        {
            return otherPair.GetName() == pair.GetName();
        });

        if (includePair)
        {
            m_Pairs.push_back(otherPair);
        }
    }
}

bool KernelConfiguration::operator==(const KernelConfiguration& other) const
{
    const auto& pairs = GetPairs();
    const auto& otherPairs = other.GetPairs();

    if (pairs.size() != otherPairs.size())
    {
        return false;
    }

    for (const auto& pair : pairs)
    {
        const bool hasPair = ContainsElementIf(otherPairs, [&pair](const auto& otherPair)
        {
            return pair.GetName() == otherPair.GetName();
        });

        if (!hasPair)
        {
            return false;
        }
    }

    bool matchingValues = true;

    for (const auto& pair : pairs)
    {
        for (const auto& otherPair : otherPairs)
        {
            if (pair.GetName() == otherPair.GetName())
            {
                matchingValues &= pair.HasSameValue(otherPair);
                break;
            }
        }
    }

    return matchingValues;
}

bool KernelConfiguration::operator!=(const KernelConfiguration& other) const
{
    return !(*this == other);
}

} // namespace ktt
