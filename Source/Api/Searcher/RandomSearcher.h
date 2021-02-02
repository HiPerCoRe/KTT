/** @file RandomSearcher.h
  * Searcher which explores configurations in random order.
  */
#pragma once

#include <cstddef>

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
    void OnReset() override;

    void CalculateNextConfiguration(const KernelResult& previousResult) override;
    const KernelConfiguration& GetCurrentConfiguration() const override;
    uint64_t GetUnexploredConfigurationCount() const override;

private:
    size_t m_Index;
    std::vector<size_t> m_ConfigurationIndices;
};

} // namespace ktt
