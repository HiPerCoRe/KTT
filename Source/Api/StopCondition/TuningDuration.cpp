#include <algorithm>

#include <Api/StopCondition/TuningDuration.h>

namespace ktt
{

TuningDuration::TuningDuration(const double duration) :
    m_PassedTime(0.0),
    m_TargetTime(std::max(0.0, duration))
{}

bool TuningDuration::IsFulfilled() const
{
    return m_PassedTime > m_TargetTime;
}

void TuningDuration::Initialize()
{
    m_InitialTime = std::chrono::steady_clock::now();
    m_PassedTime = 0.0;
}

void TuningDuration::Update([[maybe_unused]] const KernelResult& result)
{
    const auto currentTime = std::chrono::steady_clock::now();
    m_PassedTime = static_cast<double>(std::chrono::duration_cast<std::chrono::seconds>(currentTime - m_InitialTime).count());
}

std::string TuningDuration::GetStatusString() const
{
    return "Current tuning time: " + std::to_string(m_PassedTime) + " / " + std::to_string(m_TargetTime) + " seconds";
}

} // namespace ktt
