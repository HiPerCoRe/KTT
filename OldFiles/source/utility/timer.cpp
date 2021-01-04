#include <utility/timer.h>

namespace ktt
{

void Timer::start()
{
    initialTime = std::chrono::steady_clock::now();
}

void Timer::stop()
{
    endTime = std::chrono::steady_clock::now();
}

uint64_t Timer::getElapsedTime() const
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - initialTime).count());
}

} // namespace ktt
