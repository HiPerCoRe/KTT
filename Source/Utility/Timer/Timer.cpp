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

} // namespace ktt
