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
        Searcher(),
        index(0)
    {}

    void onReset() override
    {
        index = 0;
    }

    void calculateNextConfiguration(const ComputationResult&) override
    {
        ++index;
    }

    const KernelConfiguration& getNextConfiguration() const override
    {
        return getConfigurations().at(index);
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
};

} // namespace ktt
