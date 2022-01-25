#ifdef KTT_PYTHON

#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Ktt.h>

namespace py = pybind11;

void InitializePythonComputeInterface(py::module_& module)
{
    py::class_<ktt::ComputeInterface>(module, "ComputeInterface")
        .def("RunKernel", py::overload_cast<const ktt::KernelDefinitionId>(&ktt::ComputeInterface::RunKernel))
        .def("RunKernel", py::overload_cast<const ktt::KernelDefinitionId, const ktt::DimensionVector&,
            const ktt::DimensionVector&>(&ktt::ComputeInterface::RunKernel))
        .def("RunKernelAsync", py::overload_cast<const ktt::KernelDefinitionId, const ktt::QueueId>(&ktt::ComputeInterface::RunKernelAsync))
        .def("RunKernelAsync", py::overload_cast<const ktt::KernelDefinitionId, const ktt::QueueId, const ktt::DimensionVector&,
            const ktt::DimensionVector&>(&ktt::ComputeInterface::RunKernelAsync))
        .def("WaitForComputeAction", &ktt::ComputeInterface::WaitForComputeAction)
        .def("RunKernelWithProfiling", py::overload_cast<const ktt::KernelDefinitionId>(&ktt::ComputeInterface::RunKernelWithProfiling))
        .def("RunKernelWithProfiling", py::overload_cast<const ktt::KernelDefinitionId, const ktt::DimensionVector&,
            const ktt::DimensionVector&>(&ktt::ComputeInterface::RunKernelWithProfiling))
        .def("GetRemainingProfilingRuns", [](ktt::ComputeInterface& ci, const ktt::KernelDefinitionId id) { return ci.GetRemainingProfilingRuns(id); })
        .def("GetRemainingProfilingRuns", [](ktt::ComputeInterface& ci) { return ci.GetRemainingProfilingRuns(); })
        .def("GetDefaultQueue", &ktt::ComputeInterface::GetDefaultQueue)
        .def("GetAllQueues", &ktt::ComputeInterface::GetAllQueues)
        .def("SynchronizeQueue", &ktt::ComputeInterface::SynchronizeQueue)
        .def("SynchronizeQueues", &ktt::ComputeInterface::SynchronizeQueues)
        .def("SynchronizeDevice", &ktt::ComputeInterface::SynchronizeDevice)
        .def("GetCurrentGlobalSize", &ktt::ComputeInterface::GetCurrentGlobalSize, py::return_value_policy::reference)
        .def("GetCurrentLocalSize", &ktt::ComputeInterface::GetCurrentLocalSize, py::return_value_policy::reference)
        .def("GetCurrentConfiguration", &ktt::ComputeInterface::GetCurrentConfiguration, py::return_value_policy::reference)
        .def("GetRunMode", &ktt::ComputeInterface::GetRunMode)
        .def("ChangeArguments", &ktt::ComputeInterface::ChangeArguments)
        .def("SwapArguments", &ktt::ComputeInterface::SwapArguments)
        .def("UpdateScalarArgumentChar", [](ktt::ComputeInterface& ci, const ktt::ArgumentId id, const int8_t data) { ci.UpdateScalarArgument(id, &data); })
        .def("UpdateScalarArgumentShort", [](ktt::ComputeInterface& ci, const ktt::ArgumentId id, const int16_t data) { ci.UpdateScalarArgument(id, &data); })
        .def("UpdateScalarArgumentInt", [](ktt::ComputeInterface& ci, const ktt::ArgumentId id, const int32_t data) { ci.UpdateScalarArgument(id, &data); })
        .def("UpdateScalarArgumentLong", [](ktt::ComputeInterface& ci, const ktt::ArgumentId id, const int64_t data) { ci.UpdateScalarArgument(id, &data); })
        .def("UpdateScalarArgumentFloat", [](ktt::ComputeInterface& ci, const ktt::ArgumentId id, const float data) { ci.UpdateScalarArgument(id, &data); })
        .def("UpdateScalarArgumentDouble", [](ktt::ComputeInterface& ci, const ktt::ArgumentId id, const double data) { ci.UpdateScalarArgument(id, &data); })
        .def("UpdateLocalArgument", &ktt::ComputeInterface::UpdateLocalArgument)
        .def("UploadBuffer", &ktt::ComputeInterface::UploadBuffer)
        .def("UploadBufferAsync", &ktt::ComputeInterface::UploadBufferAsync)
        .def
        (
            "DownloadBuffer",
            &ktt::ComputeInterface::DownloadBuffer,
            py::arg("id"),
            py::arg("destination"),
            py::arg("dataSize") = 0
        )
        .def
        (
            "DownloadBufferAsync",
            &ktt::ComputeInterface::DownloadBufferAsync,
            py::arg("id"),
            py::arg("queue"),
            py::arg("destination"),
            py::arg("dataSize") = 0
        )
        .def
        (
            "UpdateBuffer",
            &ktt::ComputeInterface::UpdateBuffer,
            py::arg("id"),
            py::arg("data"),
            py::arg("dataSize") = 0
        )
        .def
        (
            "UpdateBufferAsync",
            &ktt::ComputeInterface::UpdateBufferAsync,
            py::arg("id"),
            py::arg("queue"),
            py::arg("data"),
            py::arg("dataSize") = 0
        )
        .def
        (
            "CopyBuffer",
            &ktt::ComputeInterface::CopyBuffer,
            py::arg("destination"),
            py::arg("source"),
            py::arg("dataSize") = 0
        )
        .def
        (
            "CopyBufferAsync",
            &ktt::ComputeInterface::CopyBufferAsync,
            py::arg("destination"),
            py::arg("source"),
            py::arg("queue"),
            py::arg("dataSize") = 0
        )
        .def("WaitForTransferAction", &ktt::ComputeInterface::WaitForTransferAction)
        .def("ResizeBuffer", &ktt::ComputeInterface::ResizeBuffer)
        .def("ClearBuffer", &ktt::ComputeInterface::ClearBuffer)
        .def("HasBuffer", &ktt::ComputeInterface::HasBuffer)
        .def("GetUnifiedMemoryBufferHandle", &ktt::ComputeInterface::GetUnifiedMemoryBufferHandle);
}

#endif // KTT_PYTHON
