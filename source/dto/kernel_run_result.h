#pragma once

#include <cstdint>
#include <vector>

#include "../kernel_argument/kernel_argument.h"

namespace ktt
{

class KernelRunResult
{
public:
    KernelRunResult();
    explicit KernelRunResult(const uint64_t duration, const std::vector<KernelArgument>& resultArguments);

    uint64_t getDuration() const;
    std::vector<KernelArgument> getResultArguments() const;

private:
    uint64_t duration;
    std::vector<KernelArgument> resultArguments;
};

} // namespace ktt
