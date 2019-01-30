#pragma once

#include <tuning_runner/searcher/searcher.h>

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

    void calculateNextConfiguration(const bool, const KernelConfiguration&, const double, const KernelProfilingData&,
        const std::map<KernelId, KernelProfilingData>&) override
    {
        index++;
    }

    KernelConfiguration getNextConfiguration() const override
    {
        return configurations.at(index);
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
    size_t index;
};

} // namespace ktt
