/** @file McmcSearcher.h
  * Searcher which explores configurations using Markov chain Monte Carlo method.
  */
#pragma once

#include <cstddef>
#include <random>
#include <set>
#include <vector>

#include <Api/Searcher/Searcher.h>
#include <KttPlatform.h>

namespace ktt
{

/** @class McmcSearcher
  * Searcher which explores configurations using Markov chain Monte Carlo method.
  */
class KTT_API McmcSearcher : public Searcher
{
public:
    /** @fn McmcSearcher(const std::vector<double>& start)
      * Initializes MCMC searcher.
      * @param start Optional parameter which specifies starting point for MCMC searcher.
      */
    McmcSearcher(const std::vector<double>& start);

    void OnInitialize() override;
    void OnReset() override;

    bool CalculateNextConfiguration(const KernelResult& previousResult) override;
    KernelConfiguration GetCurrentConfiguration() const override;

private:
    uint64_t m_Index;
    size_t m_VisitedStatesCount;
    size_t m_OriginState;
    size_t m_CurrentState;
    size_t m_Boot;
    double m_BestTime;

    std::vector<double> m_Start;
    std::vector<double> m_ExecutionTimes;
    std::set<size_t> m_UnexploredIndices;

    std::default_random_engine m_Generator;
    std::uniform_int_distribution<size_t> m_IntDistribution;
    std::uniform_real_distribution<double> m_ProbabilityDistribution;

    inline static size_t m_MaximumDifferences = 2;
    inline static size_t m_BootIterations = 10;
    inline static double m_EscapeProbability = 0.02;

    std::vector<size_t> GetNeighbours(const size_t referenceId) const;
    size_t SearchStateIndex(const std::vector<double>& state) const;
};

} // namespace ktt
