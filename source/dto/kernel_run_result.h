#pragma once

#include <cstdint>
#include <vector>

namespace ktt
{

class KernelRunResult
{
public:
    KernelRunResult();
    explicit KernelRunResult(const uint64_t duration, const uint64_t overhead);

    void increaseOverhead(const uint64_t overhead);

    bool isValid() const;
    uint64_t getDuration() const;
    uint64_t getOverhead() const;

private:
    bool valid;
    uint64_t duration;
    uint64_t overhead;
};

} // namespace ktt
