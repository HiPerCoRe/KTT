/** @file full_searcher.h
  * Searcher which explores configurations in deterministic order.
  */
#pragma once

#include <api/searcher/searcher.h>

namespace ktt
{

/** @class FullSearcher
  * Searcher which explores configurations in deterministic order.
  */
class FullSearcher : public Searcher
{
public:
    /** @fn FullSearcher()
      * Initializes full searcher.
      */
    FullSearcher() :
        configurations(nullptr),
        index(0)
    {}

    void initializeConfigurations(const std::vector<KernelConfiguration>& configurations) override
    {
        if (configurations.empty())
        {
            throw std::runtime_error("No configurations provided for full searcher");
        }

        this->configurations = &configurations;
    }

    void calculateNextConfiguration(const ComputationResult&) override
    {
        ++index;
    }

    const KernelConfiguration& getNextConfiguration() const override
    {
        return configurations->at(index);
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
        index = 0;
    }

private:
    const std::vector<KernelConfiguration>* configurations;
    size_t index;
};

} // namespace ktt
