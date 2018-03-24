#pragma once

#include <algorithm>
#include <random>
#include "searcher.h"

namespace ktt
{

class RandomSearcher : public Searcher
{
public:
    RandomSearcher(const std::vector<KernelConfiguration>& configurations) :
        configurations(configurations),
        index(0)
    {
        if (configurations.size() == 0)
        {
            throw std::runtime_error("Configurations vector provided for searcher is empty");
        }

        std::random_device device;
        std::default_random_engine engine(device());
        std::shuffle(std::begin(this->configurations), std::end(this->configurations), engine);
    }

    void calculateNextConfiguration(const double) override
    {
        index++;
    }

    KernelConfiguration getCurrentConfiguration() const override
    {
        return configurations.at(index);
    }

    size_t getConfigurationCount() const override
    {
        return configurations.size();
    }

    size_t getUnexploredConfigurationCount() const override
    {
        return getConfigurationCount() - index;
    }

private:
    std::vector<KernelConfiguration> configurations;
    size_t index;
};

} // namespace ktt
