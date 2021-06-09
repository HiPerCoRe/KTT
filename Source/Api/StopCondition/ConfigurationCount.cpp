#include <algorithm>

#include <Api/StopCondition/ConfigurationCount.h>

namespace ktt
{

ConfigurationCount::ConfigurationCount(const uint64_t count) :
    m_CurrentCount(0),
    m_TargetCount(std::max(static_cast<uint64_t>(1), count))
{}

bool ConfigurationCount::IsFulfilled() const
{
    return m_CurrentCount >= m_TargetCount;
}

void ConfigurationCount::Initialize(const uint64_t configurationsCount)
{
    m_CurrentCount = 0;
    m_TargetCount = std::min(m_TargetCount, configurationsCount);
}

void ConfigurationCount::Update([[maybe_unused]] const KernelResult& result)
{
    ++m_CurrentCount;
}
    
std::string ConfigurationCount::GetStatusString() const
{
    return "Current count of explored configurations: " + std::to_string(m_CurrentCount) + " / " + std::to_string(m_TargetCount);
}

} // namespace ktt
