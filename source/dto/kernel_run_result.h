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
    explicit KernelRunResult(const uint64_t duration, const uint64_t overhead, const std::vector<KernelArgument>& resultArguments);

    void increaseOverhead(const uint64_t overhead);
    bool isValid() const;

    uint64_t getDuration() const;
    uint64_t getOverhead() const;
    std::vector<KernelArgument> getResultArguments() const;

private:
    bool valid;
    uint64_t duration;
    uint64_t overhead;
    std::vector<KernelArgument> resultArguments;
};

} // namespace ktt
