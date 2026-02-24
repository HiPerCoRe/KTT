#include <algorithm>

#include <Api/Output/ResultStatus.h>
#include <Api/StopCondition/FailureFraction.h>

namespace ktt
{

FailureFraction::FailureFraction(const double fraction, const u_int64_t minCount) :
    m_TotalExplored(0),
    m_Failures(0),
    m_MinCount(minCount)
{
    m_TargetFraction = std::clamp(fraction, 0.0, 1.0);
}

bool FailureFraction::IsFulfilled() const
{
    return (m_TotalExplored >= m_MinCount) && (GetFailureFraction() >= m_TargetFraction);
}

void FailureFraction::Initialize([[maybe_unused]] const uint64_t configurationsCount)
{
    m_TotalExplored = 0;
    m_Failures = 0;
}

void FailureFraction::Update(const KernelResult& result)
{
    ++m_TotalExplored;
    if (result.GetStatus() != ResultStatus::Ok)
    {
        ++m_Failures;
    }
}

std::string FailureFraction::GetStatusString() const
{
    return "Current fraction of failed kernel runs: " + std::to_string(GetFailureFraction()) + " / "
        + std::to_string(m_TargetFraction);
}

double FailureFraction::GetFailureFraction() const
{
    if (m_TotalExplored == 0)
    {
        return 0.0;
    }
    return static_cast<double>(m_Failures) / static_cast<double>(m_TotalExplored);
}

} // namespace ktt