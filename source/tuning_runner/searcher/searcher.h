#pragma once

#include <dto/kernel_result.h>

namespace ktt
{

class Searcher
{
public:
    virtual ~Searcher() = default;
    virtual void calculateNextConfiguration(const KernelResult& previousResult) = 0;
    virtual KernelConfiguration getNextConfiguration() const = 0;
    virtual size_t getUnexploredConfigurationCount() const = 0;
};

} // namespace ktt
