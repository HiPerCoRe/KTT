/** @file RandomSearcher.h
  * Searcher which explores configurations in random order.
  */
#pragma once

//#include <pybind11/pybind11.h>

#include <Api/Searcher/Searcher.h>
#include <KttPlatform.h>

//namespace py = pybind11;

namespace ktt
{

/** @class RandomSearcher
  * Searcher which explores configurations in random order.
  */
class KTT_API ProfileBasedSearcher : public Searcher
{
public:
    /** @fn RandomSearcher()
      * Initializes random searcher.
       */
    ProfileBasedSearcher();

    void OnInitialize() override;

    bool CalculateNextConfiguration(const KernelResult& previousResult) override;
    KernelConfiguration GetCurrentConfiguration() const override;

private:
    KernelConfiguration m_CurrentConfiguration;
    //py::object m_PythonImplementation;
    void *m_PythonImplementation;
};

} // namespace ktt
