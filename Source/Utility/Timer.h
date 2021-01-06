#pragma once

#include <chrono>
#include <cstdint>

namespace ktt
{

class Timer
{
public:
    Timer();

    void Start();
    void Stop();

    uint64_t GetElapsedTime() const;

private:
    std::chrono::steady_clock::time_point m_InitialTime;
    std::chrono::steady_clock::time_point m_EndTime;
    bool m_Running;
};

} // namespace ktt
