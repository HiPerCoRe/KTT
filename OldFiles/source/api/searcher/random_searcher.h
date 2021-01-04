/** @file random_searcher.h
  * Searcher which explores configurations in random order.
  */
#pragma once

#include <algorithm>
#include <random>
#include <api/searcher/searcher.h>

namespace ktt
{

/** @class RandomSearcher
  * Searcher which explores configurations in random order.
  */
class RandomSearcher : public Searcher
{
public:
    /** @fn RandomSearcher()
      * Initializes random searcher.
       */
    RandomSearcher() :
        Searcher(),
        index(0)
    {}

    void onInitialize() override
    {
        configurationIndices.resize(getConfigurations().size());

        for (size_t i = 0; i < configurationIndices.size(); ++i)
        {
            configurationIndices[i] = i;
        }

        std::random_device device;
        std::default_random_engine engine(device());
        std::shuffle(std::begin(configurationIndices), std::end(configurationIndices), engine);
    }

    void onReset() override
    {
        index = 0;
        configurationIndices.clear();
    }

    void calculateNextConfiguration(const ComputationResult&) override
    {
        ++index;
    }

    const KernelConfiguration& getNextConfiguration() const override
    {
        const size_t currentIndex = configurationIndices.at(index);
        return getConfigurations().at(currentIndex);
    }

    size_t getUnexploredConfigurationCount() const override
    {
        if (index >= getConfigurations().size())
        {
            return 0;
        }

        return getConfigurations().size() - index;
    }

private:
    size_t index;
    std::vector<size_t> configurationIndices;
};

} // namespace ktt
