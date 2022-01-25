#ifdef KTT_PYTHON

#include <pybind11/pybind11.h>

#include <Ktt.h>

namespace py = pybind11;

void InitializePythonEnums(py::module_& module);
void InitializePythonDataHolders(py::module_& module);
void InitializePythonSearchers(py::module_& module);
void InitializePythonStopConditions(py::module_& module);
void InitializePythonComputeInterface(py::module_& module);
void InitializePythonTuner(py::module_& module);

PYBIND11_MODULE(pyktt, module)
{
    module.doc() = "Python bindings for KTT auto-tuning framework (https://github.com/HiPerCoRe/KTT)";

    module.attr("KTT_VERSION_MAJOR") = KTT_VERSION_MAJOR;
    module.attr("KTT_VERSION_MINOR") = KTT_VERSION_MINOR;
    module.attr("KTT_VERSION_PATCH") = KTT_VERSION_PATCH;

    module.def("GetKttVersion", &ktt::GetKttVersion);
    module.def("GetKttVersionString", &ktt::GetKttVersionString);

    module.attr("InvalidQueueId") = ktt::InvalidQueueId;
    module.attr("InvalidKernelDefinitionId") = ktt::InvalidKernelDefinitionId;
    module.attr("InvalidKernelId") = ktt::InvalidKernelId;
    module.attr("InvalidArgumentId") = ktt::InvalidArgumentId;
    module.attr("InvalidDuration") = ktt::InvalidDuration;

    InitializePythonEnums(module);
    InitializePythonDataHolders(module);
    InitializePythonSearchers(module);
    InitializePythonStopConditions(module);

    py::register_exception<ktt::KttException>(module, "KttException", PyExc_Exception);

    InitializePythonComputeInterface(module);
    InitializePythonTuner(module);
}

#endif // KTT_PYTHON
