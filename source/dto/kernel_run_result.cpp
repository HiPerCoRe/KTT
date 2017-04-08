#include "kernel_run_result.h"

namespace ktt
{

KernelRunResult::KernelRunResult():
    duration(0)
{}

KernelRunResult::KernelRunResult(const uint64_t duration, const std::vector<KernelArgument>& resultArguments):
    duration(duration),
    resultArguments(resultArguments)
{}

uint64_t KernelRunResult::getDuration() const
{
    return duration;
}

std::vector<KernelArgument> KernelRunResult::getResultArguments() const
{
    return resultArguments;
}

} // namespace ktt
