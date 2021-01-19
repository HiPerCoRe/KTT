#include <ComputeEngine/TransferResult.h>
#include <Utility/ErrorHandling/Assert.h>

namespace ktt
{

TransferResult::TransferResult() :
    m_Duration(InvalidDuration),
    m_Overhead(InvalidDuration)
{}

TransferResult::TransferResult(const Nanoseconds duration, const Nanoseconds overhead) :
    m_Duration(duration),
    m_Overhead(overhead)
{}

void TransferResult::SetDuration(const Nanoseconds duration)
{
    m_Duration = duration;
}

void TransferResult::SetOverhead(const Nanoseconds overhead)
{
    m_Overhead = overhead;
}

Nanoseconds TransferResult::GetDuration() const
{
    return m_Duration;
}

Nanoseconds TransferResult::GetOverhead() const
{
    return m_Overhead;
}

} // namespace ktt
