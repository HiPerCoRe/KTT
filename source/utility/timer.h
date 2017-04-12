#pragma once

#include <chrono>
#include <cstdint>

namespace ktt
{

class Timer
{
public:
    void start();
    void stop();
    uint64_t getElapsedTime() const;

private:
    std::chrono::high_resolution_clock::time_point initialTime;
    std::chrono::high_resolution_clock::time_point endTime;
};

} // namespace ktt
