#include <algorithm>
#include <limits>

#include <Api/StopCondition/ConfigurationDuration.h>

namespace ktt
{

ConfigurationDuration::ConfigurationDuration(const double duration) :
    m_BestDuration(std::numeric_limits<double>::max()),
    m_TargetDuration(std::max(0.0, duration))
{}

bool ConfigurationDuration::IsFulfilled() const
{
    return m_BestDuration <= m_TargetDuration;
}

void ConfigurationDuration::Initialize()
{
    m_BestDuration = std::numeric_limits<double>::max();
}

void ConfigurationDuration::Update(const KernelResult& result)
{
    if (result.IsValid())
    {
        m_BestDuration = std::min(m_BestDuration, static_cast<double>(result.GetTotalDuration()) / 1'000'000.0);
    }
}

std::string ConfigurationDuration::GetStatusString() const
{
    return "Current best known configuration duration: " + std::to_string(m_BestDuration) + "ms, target duration: "
        + std::to_string(m_TargetDuration) + "ms";
}

} // namespace ktt
