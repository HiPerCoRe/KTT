#pragma once

#include <algorithm>
#include <random>
#include "searcher.h"

namespace ktt
{

class RandomSearcher : public Searcher
{
public:
    RandomSearcher(const std::vector<KernelConfiguration>& configurations, const double fraction) :
        configurations(configurations),
        index(0),
        fraction(fraction)
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
        return std::max(static_cast<size_t>(1), std::min(configurations.size(), static_cast<size_t>(configurations.size() * fraction)));
    }

    size_t getUnexploredConfigurationCount() const override
    {
        return getConfigurationCount() - index;
    }

private:
    std::vector<KernelConfiguration> configurations;
    size_t index;
    double fraction;
};

} // namespace ktt
