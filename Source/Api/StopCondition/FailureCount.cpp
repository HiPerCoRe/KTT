#include <algorithm>

#include <Api/Output/ResultStatus.h>
#include <Api/StopCondition/FailureCount.h>

namespace ktt
{

FailureCount::FailureCount(const uint64_t maxFailures) :
    m_CurrentFailures(0),
    m_MaxFailures(std::max(static_cast<uint64_t>(1), maxFailures))
{}

bool FailureCount::IsFulfilled() const
{
    return m_CurrentFailures >= m_MaxFailures;
}

void FailureCount::Initialize([[maybe_unused]] const uint64_t configurationsCount)
{
    m_CurrentFailures = 0;
}

void FailureCount::Update(const KernelResult& result)
{
    if (result.GetStatus() != ResultStatus::Ok)
    {
        ++m_CurrentFailures;
    }
}

std::string FailureCount::GetStatusString() const
{
    return "Current count of failed kernel runs: " + std::to_string(m_CurrentFailures) + " / " + std::to_string(m_MaxFailures);
}

} // namespace ktt