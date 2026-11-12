/** @file RandomSearcher.h
  * Searcher which explores configurations in random order.
  */
#pragma once

#include <cstdint>

#include <Api/Searcher/Searcher.h>
#include <KttPlatform.h>

namespace ktt
{

/** @class RandomSearcher
  * Searcher which explores configurations in random order.
  */
class KTT_API RandomSearcher : public Searcher
{
public:
    /** @fn RandomSearcher()
      * Initializes random searcher. Configurations are explored in a different order in each tuning run.
       */
    RandomSearcher();

    /** @fn explicit RandomSearcher(const uint64_t seed)
      * Initializes random searcher with the specified seed. The searcher explores configurations in the same order in
      * every tuning run, provided that tuning parameters, order of their addition and their values were not changed.
      * @param seed Seed for the source of randomness used by the searcher. See Searcher::SetSeed method for more
      * information.
      */
    explicit RandomSearcher(const uint64_t seed);

    void OnInitialize() override;

    bool CalculateNextConfiguration(const KernelResult& previousResult) override;
    KernelConfiguration GetCurrentConfiguration() const override;

private:
    KernelConfiguration m_CurrentConfiguration;
};

} // namespace ktt
