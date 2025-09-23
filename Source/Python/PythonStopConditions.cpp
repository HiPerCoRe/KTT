#ifdef KTT_PYTHON

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Ktt.h>

namespace py = pybind11;

class KTT_VISIBILITY_HIDDEN PyStopCondition : public ktt::StopCondition, public py::trampoline_self_life_support
{
public:
    using StopCondition::StopCondition;

    bool IsFulfilled() const override
    {
        PYBIND11_OVERRIDE_PURE(bool, ktt::StopCondition, IsFulfilled);
    }

    void Initialize(const uint64_t configurationsCount) override
    {
        PYBIND11_OVERRIDE_PURE(void, ktt::StopCondition, Initialize, configurationsCount);
    }

    void Update(const ktt::KernelResult& result) override
    {
        PYBIND11_OVERRIDE_PURE(void, ktt::StopCondition, Update, result);
    }

    std::string GetStatusString() const override
    {
        PYBIND11_OVERRIDE_PURE(std::string, ktt::StopCondition, GetStatusString);
    }
};

void InitializePythonStopConditions(py::module_& module)
{
    py::class_<ktt::StopCondition, PyStopCondition, py::smart_holder>(module, "StopCondition")
        .def(py::init<>())
        .def("IsFulfilled", &ktt::StopCondition::IsFulfilled)
        .def("Initialize", &ktt::StopCondition::Initialize)
        .def("Update", &ktt::StopCondition::Update)
        .def("GetStatusString", &ktt::StopCondition::GetStatusString);

    py::class_<ktt::ConfigurationCount, ktt::StopCondition, py::smart_holder>(module, "ConfigurationCount")
        .def(py::init<const uint64_t>());

    py::class_<ktt::ConfigurationDuration, ktt::StopCondition, py::smart_holder>(module, "ConfigurationDuration")
        .def(py::init<const double>());

    py::class_<ktt::ConfigurationFraction, ktt::StopCondition, py::smart_holder>(module, "ConfigurationFraction")
        .def(py::init<const double>());

    py::class_<ktt::TuningDuration, ktt::StopCondition, py::smart_holder>(module, "TuningDuration")
        .def(py::init<const double>());
}

#endif // KTT_PYTHON
