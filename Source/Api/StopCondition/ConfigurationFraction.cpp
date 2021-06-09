#include <algorithm>

#include <Api/StopCondition/ConfigurationFraction.h>

namespace ktt
{

ConfigurationFraction::ConfigurationFraction(const double fraction) :
    m_CurrentCount(0),
    m_TotalCount(0)
{
    m_TargetFraction = std::clamp(fraction, 0.0, 1.0);
}

bool ConfigurationFraction::IsFulfilled() const
{
    return GetExploredFraction() >= m_TargetFraction;
}

void ConfigurationFraction::Initialize(const uint64_t configurationsCount)
{
    m_CurrentCount = 0;
    m_TotalCount = std::max(static_cast<uint64_t>(1), configurationsCount);
}

void ConfigurationFraction::Update([[maybe_unused]] const KernelResult& result)
{
    ++m_CurrentCount;
}

std::string ConfigurationFraction::GetStatusString() const
{
    return "Current fraction of explored configurations: " + std::to_string(GetExploredFraction()) + " / "
        + std::to_string(m_TargetFraction);
}

double ConfigurationFraction::GetExploredFraction() const
{
    return static_cast<double>(m_CurrentCount) / static_cast<double>(m_TotalCount);
}

} // namespace ktt
