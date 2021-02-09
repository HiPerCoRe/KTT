#pragma once

#include <chrono>
#include <string>

#include <Output/TimeUnit.h>
#include <KttTypes.h>

namespace ktt
{

class Timer
{
public:
    Timer();

    void Start();
    void Stop();

    Nanoseconds GetElapsedTime() const;

    static uint64_t ConvertDuration(const Nanoseconds duration, const TimeUnit unit);
    static std::string GetTag(const TimeUnit unit);

private:
    std::chrono::steady_clock::time_point m_InitialTime;
    std::chrono::steady_clock::time_point m_EndTime;
    bool m_Running;
};

} // namespace ktt
