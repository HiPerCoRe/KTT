#pragma once

#include "searcher.h"

namespace ktt
{

class FullSearcher : public Searcher
{
public:
    FullSearcher(const std::vector<KernelConfiguration>& configurations) :
        configurations(configurations),
        index(0)
    {
        if (configurations.size() == 0)
        {
            throw std::runtime_error("Configurations vector provided for searcher is empty");
        }
    }

    KernelConfiguration getNextConfiguration() override
    {
        return configurations.at(index);
    }

    void calculateNextConfiguration(const double previousConfigurationDuration) override
    {
        index++;
    }

    size_t getConfigurationsCount() const override
    {
        return configurations.size();
    }

private:
    std::vector<KernelConfiguration> configurations;
    size_t index;
};

} // namespace ktt
