/** @file DeterministicSearcher.h
  * Searcher which explores configurations in deterministic order.
  */
#pragma once

#include <cstddef>

#include <Api/Searcher/Searcher.h>
#include <KttPlatform.h>

namespace ktt
{

/** @class DeterministicSearcher
  * Searcher which explores configurations in deterministic order.
  */
class KTT_API DeterministicSearcher : public Searcher
{
public:
    /** @fn DeterministicSearcher()
      * Initializes deterministic searcher.
      */
    DeterministicSearcher();

    void OnReset() override;

    void CalculateNextConfiguration(const KernelResult& previousResult) override;
    const KernelConfiguration& GetCurrentConfiguration() const override;

private:
    size_t m_Index;
};

} // namespace ktt
