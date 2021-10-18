#ifdef KTT_PYTHON

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Ktt.h>

namespace py = pybind11;

PYBIND11_MODULE(ktt, module)
{
    module.doc() = "Python bindings for KTT auto-tuning framework (https://github.com/HiPerCoRe/KTT)";

    module.attr("InvalidQueueId") = ktt::InvalidQueueId;
    module.attr("InvalidKernelDefinitionId") = ktt::InvalidKernelDefinitionId;
    module.attr("InvalidKernelId") = ktt::InvalidKernelId;
    module.attr("InvalidArgumentId") = ktt::InvalidArgumentId;
    module.attr("InvalidDuration") = ktt::InvalidDuration;

    module.def("GetKttVersion", &ktt::GetKttVersion, "Returns the current KTT framework version in integer format.");
    module.def("GetKttVersionString", &ktt::GetKttVersionString, "Returns the current KTT framework version in string format.");

    py::enum_<ktt::ComputeApi>(module, "ComputeApi")
        .value("OpenCL", ktt::ComputeApi::OpenCL)
        .value("CUDA", ktt::ComputeApi::CUDA)
        .value("Vulkan", ktt::ComputeApi::Vulkan);

    py::enum_<ktt::ModifierAction>(module, "ModifierAction")
        .value("Add", ktt::ModifierAction::Add)
        .value("Subtract", ktt::ModifierAction::Subtract)
        .value("Multiply", ktt::ModifierAction::Multiply)
        .value("Divide", ktt::ModifierAction::Divide)
        .value("DivideCeil", ktt::ModifierAction::DivideCeil);

    py::enum_<ktt::ModifierDimension>(module, "ModifierDimension")
        .value("X", ktt::ModifierDimension::X)
        .value("Y", ktt::ModifierDimension::Y)
        .value("Z", ktt::ModifierDimension::Z);

    py::class_<ktt::DimensionVector>(module, "DimensionVector")
        .def(py::init<>())
        .def(py::init<const size_t>())
        .def(py::init<const size_t, const size_t>())
        .def(py::init<const size_t, const size_t, const size_t>())
        .def(py::init<const std::vector<size_t>&>())
        .def("SetSizeX", &ktt::DimensionVector::SetSizeX)
        .def("SetSizeY", &ktt::DimensionVector::SetSizeY)
        .def("SetSizeZ", &ktt::DimensionVector::SetSizeZ)
        .def("SetSize", &ktt::DimensionVector::SetSize)
        .def("Multiply", &ktt::DimensionVector::Multiply)
        .def("Divide", &ktt::DimensionVector::Divide)
        .def("RoundUp", &ktt::DimensionVector::RoundUp)
        .def("ModifyByValue", &ktt::DimensionVector::ModifyByValue)
        .def("GetSizeX", &ktt::DimensionVector::GetSizeX)
        .def("GetSizeY", &ktt::DimensionVector::GetSizeY)
        .def("GetSizeZ", &ktt::DimensionVector::GetSizeZ)
        .def("GetSize", &ktt::DimensionVector::GetSize)
        .def("GetTotalSize", &ktt::DimensionVector::GetTotalSize)
        .def("GetVector", &ktt::DimensionVector::GetVector)
        .def("GetString", &ktt::DimensionVector::GetString)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("__repr__", &ktt::DimensionVector::GetString);

    py::class_<ktt::Tuner>(module, "Tuner")
        .def(py::init<const ktt::PlatformIndex, const ktt::DeviceIndex, const ktt::ComputeApi>())
        .def("RemoveKernelDefinition", &ktt::Tuner::RemoveKernelDefinition);
}

#endif // KTT_PYTHON
