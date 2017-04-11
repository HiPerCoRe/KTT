#include "kernel_run_result.h"

namespace ktt
{

KernelRunResult::KernelRunResult() :
    valid(false),
    duration(UINT64_MAX)
{}

KernelRunResult::KernelRunResult(const uint64_t duration, const std::vector<KernelArgument>& resultArguments) :
    valid(true),
    duration(duration),
    resultArguments(resultArguments)
{}

bool KernelRunResult::isValid() const
{
    return valid;
}

uint64_t KernelRunResult::getDuration() const
{
    return duration;
}

std::vector<KernelArgument> KernelRunResult::getResultArguments() const
{
    return resultArguments;
}

} // namespace ktt
