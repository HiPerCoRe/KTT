#pragma once

#include <chrono>

#include <KttTypes.h>

namespace ktt
{

class Timer
{
public:
    Timer();

    void Start();
    void Stop();
    void Restart();

    Nanoseconds GetElapsedTime() const;
    Nanoseconds GetCheckpointTime() const;

private:
    std::chrono::steady_clock::time_point m_InitialTime;
    std::chrono::steady_clock::time_point m_EndTime;
    bool m_Running;
};

} // namespace ktt
