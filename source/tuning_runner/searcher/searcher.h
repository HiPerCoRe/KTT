#pragma once

#include "kernel/kernel_configuration.h"

namespace ktt
{

class Searcher
{
public:
    virtual ~Searcher() = default;
    virtual void calculateNextConfiguration(const double previousDuration) = 0;
    virtual KernelConfiguration getCurrentConfiguration() const = 0;
    virtual size_t getUnexploredConfigurationCount() const = 0;
};

} // namespace ktt
