/** @file ProfileBasedSearcher.h
  * Searcher which explores configurations according to observed bottlenecks
  * and ML model created on historical data (on the same tuning space, but 
  * possibly different HW and input size). For more information, see
  * J. Filipovic et al. Using hardware performance counters to speed up 
  * autotuning convergence on GPUs. JPDC, vol. 160, 2021.
  */
#pragma once

#include <Api/Searcher/Searcher.h>
#include <KttPlatform.h>

namespace ktt
{

/** @class ProfileBasedSearcher
  * Searcher which explores configurations leveranging performance counters.
  */
class KTT_API ProfileBasedSearcher : public Searcher
{
public:
    /** @fn ProfileBasedSearcher()
      * Initializes profile based searcher.
       */
    ProfileBasedSearcher(void* tuner, KernelId kernel, std::string model);

    void OnInitialize() override;

    bool CalculateNextConfiguration(const KernelResult& previousResult) override;
    KernelConfiguration GetCurrentConfiguration() const override;

private:
    KernelConfiguration m_CurrentConfiguration;
    void *m_PythonImplementation;
    void *m_PythonInstance;
};

} // namespace ktt
