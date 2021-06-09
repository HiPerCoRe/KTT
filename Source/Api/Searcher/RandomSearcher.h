/** @file RandomSearcher.h
  * Searcher which explores configurations in random order.
  */
#pragma once

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
      * Initializes random searcher.
       */
    RandomSearcher();

    void OnInitialize() override;

    bool CalculateNextConfiguration(const KernelResult& previousResult) override;
    KernelConfiguration GetCurrentConfiguration() const override;

private:
    KernelConfiguration m_CurrentConfiguration;
};

} // namespace ktt
