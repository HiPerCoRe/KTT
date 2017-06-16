#include "kernel_run_result.h"

namespace ktt
{

KernelRunResult::KernelRunResult() :
    valid(false),
    duration(UINT64_MAX),
    overhead(0)
{}

KernelRunResult::KernelRunResult(const uint64_t duration, const uint64_t overhead) :
    valid(true),
    duration(duration),
    overhead(overhead)
{}

void KernelRunResult::increaseOverhead(const uint64_t overhead)
{
    this->overhead += overhead;
}

bool KernelRunResult::isValid() const
{
    return valid;
}

uint64_t KernelRunResult::getDuration() const
{
    return duration;
}

uint64_t KernelRunResult::getOverhead() const
{
    return overhead;
}

} // namespace ktt
