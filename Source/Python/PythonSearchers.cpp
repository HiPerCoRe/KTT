#ifdef KTT_PYTHON

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Ktt.h>

namespace py = pybind11;

class KTT_VISIBILITY_HIDDEN PySearcher : public ktt::Searcher, public py::trampoline_self_life_support
{
public:
    using Searcher::Searcher;

    void OnInitialize() override
    {
        PYBIND11_OVERRIDE(void, ktt::Searcher, OnInitialize);
    }

    void OnReset() override
    {
        PYBIND11_OVERRIDE(void, ktt::Searcher, OnReset);
    }

    bool CalculateNextConfiguration(const ktt::KernelResult& previousResult) override
    {
        PYBIND11_OVERRIDE_PURE(bool, ktt::Searcher, CalculateNextConfiguration, previousResult);
    }

    ktt::KernelConfiguration GetCurrentConfiguration() const override
    {
        PYBIND11_OVERRIDE_PURE(ktt::KernelConfiguration, ktt::Searcher, GetCurrentConfiguration);
    }
};

void InitializePythonSearchers(py::module_& module)
{
    py::class_<ktt::Searcher, PySearcher>(module, "Searcher")
        .def(py::init<>())
        .def("OnInitialize", &ktt::Searcher::OnInitialize)
        .def("OnReset", &ktt::Searcher::OnReset)
        .def("CalculateNextConfiguration", &ktt::Searcher::CalculateNextConfiguration)
        .def("GetCurrentConfiguration", &ktt::Searcher::GetCurrentConfiguration)
        .def("GetIndex", &ktt::Searcher::GetIndex)
        .def("GetConfiguration", &ktt::Searcher::GetConfiguration)
        .def("GetRandomConfiguration", &ktt::Searcher::GetRandomConfiguration)
        .def
        (
            "GetNeighbourConfigurations",
            &ktt::Searcher::GetNeighbourConfigurations,
            py::arg("configuration"),
            py::arg("maxDifferences"),
            py::arg("maxNeighbours") = 3
        )
        .def("GetConfigurationsCount", &ktt::Searcher::GetConfigurationsCount)
        .def("GetExploredIndices", &ktt::Searcher::GetExploredIndices, py::return_value_policy::reference)
        .def("IsInitialized", &ktt::Searcher::IsInitialized);

    py::class_<ktt::DeterministicSearcher, ktt::Searcher>(module, "DeterministicSearcher")
        .def(py::init<>());

    py::class_<ktt::McmcSearcher, ktt::Searcher>(module, "McmcSearcher")
        .def(py::init<const ktt::KernelConfiguration&>(), py::arg("start") = ktt::KernelConfiguration());

    py::class_<ktt::RandomSearcher, ktt::Searcher>(module, "RandomSearcher")
        .def(py::init<>());
}

#endif // KTT_PYTHON
