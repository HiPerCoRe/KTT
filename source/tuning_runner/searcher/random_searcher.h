#pragma once

#include <algorithm>
#include <random>
#include <tuning_runner/searcher/searcher.h>

namespace ktt
{

class RandomSearcher : public Searcher
{
public:
    RandomSearcher(const std::vector<KernelConfiguration>& configurations) :
        configurations(configurations),
        configurationIndices(configurations.size()),
        index(0)
    {
        if (configurations.size() == 0)
        {
            throw std::runtime_error("Configurations vector provided for searcher is empty");
        }

        for (size_t i = 0; i < configurations.size(); i++)
        {
            configurationIndices.at(i) = i;
        }

        std::random_device device;
        std::default_random_engine engine(device());
        std::shuffle(std::begin(this->configurationIndices), std::end(this->configurationIndices), engine);
    }

    void calculateNextConfiguration(const bool, const KernelConfiguration&, const double, const KernelProfilingData&,
        const std::map<KernelId, KernelProfilingData>&) override
    {
        index++;
    }

    KernelConfiguration getNextConfiguration() const override
    {
        size_t currentIndex = configurationIndices.at(index);
        return configurations.at(currentIndex);
    }

    size_t getUnexploredConfigurationCount() const override
    {
        if (index >= configurations.size())
        {
            return 0;
        }

        return configurations.size() - index;
    }

private:
    const std::vector<KernelConfiguration>& configurations;
    std::vector<size_t> configurationIndices;
    size_t index;
};

} // namespace ktt
