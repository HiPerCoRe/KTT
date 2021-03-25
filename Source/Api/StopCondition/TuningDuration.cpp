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

void TuningDuration::Initialize([[maybe_unused]] const uint64_t configurationsCount)
{
    m_InitialTime = std::chrono::steady_clock::now();
    m_PassedTime = 0.0;
}

void TuningDuration::Update([[maybe_unused]] const KernelResult& result)
{
    const auto currentTime = std::chrono::steady_clock::now();
    const uint64_t passedMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - m_InitialTime).count();
    m_PassedTime = static_cast<double>(passedMilliseconds) / 1'000.0;
}

std::string TuningDuration::GetStatusString() const
{
    return "Current tuning time: " + std::to_string(m_PassedTime) + " / " + std::to_string(m_TargetTime) + " seconds";
}

} // namespace ktt
