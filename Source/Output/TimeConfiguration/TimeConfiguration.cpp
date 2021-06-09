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

TimeUnit TimeConfiguration::GetTimeUnit() const
{
    return m_TimeUnit;
}

std::string TimeConfiguration::GetUnitTag() const
{
    switch (m_TimeUnit)
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

uint64_t TimeConfiguration::ConvertFromNanoseconds(const Nanoseconds duration) const
{
    switch (m_TimeUnit)
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

double TimeConfiguration::ConvertFromNanosecondsDouble(const Nanoseconds duration) const
{
    switch (m_TimeUnit)
    {
    case TimeUnit::Nanoseconds:
        return static_cast<double>(duration);
    case TimeUnit::Microseconds:
        return static_cast<double>(duration) / 1'000.0;
    case TimeUnit::Milliseconds:
        return static_cast<double>(duration) / 1'000'000.0;
    case TimeUnit::Seconds:
        return static_cast<double>(duration) / 1'000'000'000.0;
    default:
        KttError("Unhandled time unit value");
        return 0.0;
    }
}

Nanoseconds TimeConfiguration::ConvertToNanoseconds(const uint64_t duration) const
{
    switch (m_TimeUnit)
    {
    case TimeUnit::Nanoseconds:
        return duration;
    case TimeUnit::Microseconds:
        return duration * 1'000;
    case TimeUnit::Milliseconds:
        return duration * 1'000'000;
    case TimeUnit::Seconds:
        return duration * 1'000'000'000;
    default:
        KttError("Unhandled time unit value");
        return 0;
    }
}

Nanoseconds TimeConfiguration::ConvertToNanosecondsDouble(const double duration) const
{
    switch (m_TimeUnit)
    {
    case TimeUnit::Nanoseconds:
        return static_cast<Nanoseconds>(duration);
    case TimeUnit::Microseconds:
        return static_cast<Nanoseconds>(duration * 1'000.0);
    case TimeUnit::Milliseconds:
        return static_cast<Nanoseconds>(duration * 1'000'000.0);
    case TimeUnit::Seconds:
        return static_cast<Nanoseconds>(duration * 1'000'000'000.0);
    default:
        KttError("Unhandled time unit value");
        return 0;
    }
}

TimeConfiguration::TimeConfiguration() :
    m_TimeUnit(TimeUnit::Milliseconds)
{}

} // namespace ktt
