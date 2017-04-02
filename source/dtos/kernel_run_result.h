#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "../kernel_argument/kernel_argument.h"

namespace ktt
{

class KernelRunResult
{
public:
    KernelRunResult():
        duration(0)
    {}

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
