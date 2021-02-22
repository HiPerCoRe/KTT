#include <Utility/ErrorHandling/Assert.h>
#include <Utility/Timer/Timer.h>

namespace ktt
{

Timer::Timer() :
    m_Running(false)
{}

void Timer::Start()
{
    KttAssert(!m_Running, "Calls to Start and Stop should be properly paired");
    m_InitialTime = std::chrono::steady_clock::now();
    m_Running = true;
}

void Timer::Stop()
{
    KttAssert(m_Running, "Calls to Start and Stop should be properly paired");
    m_EndTime = std::chrono::steady_clock::now();
    m_Running = false;
}

Nanoseconds Timer::GetElapsedTime() const
{
    KttAssert(!m_Running, "Elapsed time should only be retrieved when timer is stopped");
    return static_cast<Nanoseconds>(std::chrono::duration_cast<std::chrono::nanoseconds>(m_EndTime - m_InitialTime).count());
}

uint64_t Timer::ConvertDuration(const Nanoseconds duration, const TimeUnit unit)
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

std::string Timer::GetTag(const TimeUnit unit)
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
