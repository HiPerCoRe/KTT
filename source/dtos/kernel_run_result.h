#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "../kernel/kernel_argument.h"

namespace ktt
{

class KernelRunResult
{
public:
    explicit KernelRunResult(const uint64_t duration, const std::vector<KernelArgument>& resultArguments):
        duration(duration),
        resultArguments(resultArguments)
    {}

    uint64_t getDuration() const
    {
        return duration;
    }

    std::vector<KernelArgument> getResultArguments() const
    {
        return resultArguments;
    }

private:
    uint64_t duration;
    std::vector<KernelArgument> resultArguments;
};

} // namespace ktt
