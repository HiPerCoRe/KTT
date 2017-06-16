#pragma once

#include "kernel/kernel_configuration.h"

namespace ktt
{

class Searcher
{
public:
    virtual ~Searcher() = default;
    virtual KernelConfiguration getNextConfiguration() = 0;
    virtual void calculateNextConfiguration(const double previousConfigurationDuration) = 0;
    virtual size_t getConfigurationsCount() const = 0;
};

} // namespace ktt
