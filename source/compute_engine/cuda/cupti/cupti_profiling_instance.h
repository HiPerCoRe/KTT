#pragma once

#include <stdexcept>
#include <vector>

namespace ktt
{

class CUPTIProfilingInstance
{
public:
    explicit CUPTIProfilingInstance() :
        remainingKernelRuns(1),
        totalKernelRuns(1)
    {}

    uint64_t getRemainingKernelRuns() const
    {
        return remainingKernelRuns;
    }

    uint64_t getTotalKernelRuns() const
    {
        return totalKernelRuns;
    }

private:
    uint64_t remainingKernelRuns;
    uint64_t totalKernelRuns;
};

} // namespace ktt
