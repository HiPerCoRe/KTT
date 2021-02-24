#include <Utility/ErrorHandling/Assert.h>
#include <Output/TimeConfiguration/TimeConfiguration.h>

namespace ktt
{

TimeConfiguration& TimeConfiguration::GetInstance()
{
    static TimeConfiguration instance;
    return instance;
}

void TimeConfiguration::SetTimeUnit(const TimeUnit unit)
{
    m_TimeUnit = unit;
}

uint64_t TimeConfiguration::ConvertDuration(const Nanoseconds duration) const
{
    return ConvertDuration(duration, m_TimeUnit);
}

std::string TimeConfiguration::GetUnitTag() const
{
    return GetUnitTag(m_TimeUnit);
}

TimeConfiguration::TimeConfiguration() :
    m_TimeUnit(TimeUnit::Milliseconds)
{}

uint64_t TimeConfiguration::ConvertDuration(const Nanoseconds duration, const TimeUnit unit)
{
    switch (unit)
    {
    case TimeUnit::Nanoseconds:
        return duration;
    case TimeUnit::Microseconds:
        return duration / 1'000;
    case TimeUnit::Milliseconds:
        return duration / 1'000'000;
    case TimeUnit::Seconds:
        return duration / 1'000'000'000;
    default:
        KttError("Unhandled time unit value");
        return 0;
    }
}

std::string TimeConfiguration::GetUnitTag(const TimeUnit unit)
{
    switch (unit)
    {
    case TimeUnit::Nanoseconds:
        return "ns";
    case TimeUnit::Microseconds:
        return "us";
    case TimeUnit::Milliseconds:
        return "ms";
    case TimeUnit::Seconds:
        return "s";
    default:
        KttError("Unhandled time unit value");
        return "";
    }
}

} // namespace ktt
