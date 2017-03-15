#pragma once

#include <cstdint>
#include <string>

namespace ktt
{

class KernelRunResult
{
public:
    explicit KernelRunResult(const uint64_t duration):
        duration(duration)
    {}

    uint64_t getDuration() const
    {
        return duration;
    }

private:
    uint64_t duration;
};

} // namespace ktt
