/** @file random_searcher.h
  * Searcher which explores configurations in random order.
  */
#pragma once

#include <algorithm>
#include <random>
#include <stdexcept>
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
        configurations(nullptr),
        index(0)
    {}

    void initializeConfigurations(const std::vector<KernelConfiguration>& configurations) override
    {
        if (configurations.empty())
        {
            throw std::runtime_error("No configurations provided for random searcher");
        }

        this->configurations = &configurations;
        configurationIndices.resize(configurations.size());

        for (size_t i = 0; i < configurationIndices.size(); ++i)
        {
            configurationIndices[i] = i;
        }

        std::random_device device;
        std::default_random_engine engine(device());
        std::shuffle(std::begin(configurationIndices), std::end(configurationIndices), engine);
    }

    void calculateNextConfiguration(const ComputationResult&) override
    {
        ++index;
    }

    const KernelConfiguration& getNextConfiguration() const override
    {
        const size_t currentIndex = configurationIndices.at(index);
        return configurations->at(currentIndex);
    }

    size_t getUnexploredConfigurationCount() const override
    {
        if (index >= configurations->size())
        {
            return 0;
        }

        return configurations->size() - index;
    }

    bool isInitialized() const override
    {
        return configurations != nullptr;
    }

    void reset() override
    {
        configurations = nullptr;
        configurationIndices.clear();
        index = 0;
    }

private:
    const std::vector<KernelConfiguration>* configurations;
    std::vector<size_t> configurationIndices;
    size_t index;
};

} // namespace ktt
