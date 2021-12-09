#ifdef KTT_PYTHON

#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Ktt.h>

namespace py = pybind11;

void InitializePythonEnums(py::module_& module);
void InitializePythonDataHolders(py::module_& module);
void InitializePythonSearchers(py::module_& module);
void InitializePythonStopConditions(py::module_& module);

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

    py::class_<ktt::Tuner>(module, "Tuner")
        .def(py::init<const ktt::PlatformIndex, const ktt::DeviceIndex, const ktt::ComputeApi>())
        .def(py::init<const ktt::PlatformIndex, const ktt::DeviceIndex, const ktt::ComputeApi, const uint32_t>())
        .def
        (
            "AddKernelDefinition",
            &ktt::Tuner::AddKernelDefinition,
            py::arg("name"),
            py::arg("source"),
            py::arg("globalSize"),
            py::arg("localSize"),
            py::arg("typeNames") = std::vector<std::string>{}
        )
        .def
        (
            "AddKernelDefinitionFromFile",
            &ktt::Tuner::AddKernelDefinitionFromFile,
            py::arg("name"),
            py::arg("filePath"),
            py::arg("globalSize"),
            py::arg("localSize"),
            py::arg("typeNames") = std::vector<std::string>{}
        )
        .def
        (
            "GetKernelDefinitionId",
            &ktt::Tuner::GetKernelDefinitionId,
            py::arg("name"),
            py::arg("typeNames") = std::vector<std::string>{}
        )
        .def("RemoveKernelDefinition", &ktt::Tuner::RemoveKernelDefinition)
        .def("SetArguments", &ktt::Tuner::SetArguments)
        .def("CreateSimpleKernel", &ktt::Tuner::CreateSimpleKernel)
        .def
        (
            "CreateCompositeKernel",
            [](ktt::Tuner& tuner, const std::string& name, const std::vector<ktt::KernelDefinitionId>& definitionIds,
                std::function<void(ktt::ComputeInterface*)> launcher)
            {
                ktt::KernelLauncher actualLauncher = [launcher](ktt::ComputeInterface& interface) { launcher(&interface); };
                return tuner.CreateCompositeKernel(name, definitionIds, actualLauncher);
            },
            py::arg("name"),
            py::arg("definitionIds"),
            py::arg("launcher") = static_cast<std::function<void(ktt::ComputeInterface*)>>(nullptr)
        )
        .def("RemoveKernel", &ktt::Tuner::RemoveKernel)
        .def
        (
            "SetLauncher",
            [](ktt::Tuner& tuner, const ktt::KernelId id, std::function<void(ktt::ComputeInterface*)> launcher)
            {
                ktt::KernelLauncher actualLauncher = [launcher](ktt::ComputeInterface& interface) { launcher(&interface); };
                tuner.SetLauncher(id, actualLauncher);
            }
        )
        .def
        (
            "AddParameter",
            py::overload_cast<const ktt::KernelId, const std::string&, const std::vector<uint64_t>&, const std::string&>(&ktt::Tuner::AddParameter),
            py::arg("id"),
            py::arg("name"),
            py::arg("values"),
            py::arg("group") = std::string()
        )
        .def
        (
            "AddParameter",
            py::overload_cast<const ktt::KernelId, const std::string&, const std::vector<double>&, const std::string&>(&ktt::Tuner::AddParameter),
            py::arg("id"),
            py::arg("name"),
            py::arg("values"),
            py::arg("group") = std::string()
        )
        .def("AddThreadModifier", py::overload_cast<const ktt::KernelId, const std::vector<ktt::KernelDefinitionId>&, const ktt::ModifierType,
            const ktt::ModifierDimension, const std::vector<std::string>&, ktt::ModifierFunction>(&ktt::Tuner::AddThreadModifier))
        .def("AddThreadModifier", py::overload_cast<const ktt::KernelId, const std::vector<ktt::KernelDefinitionId>&, const ktt::ModifierType,
            const ktt::ModifierDimension, const std::string&, const ktt::ModifierAction>(&ktt::Tuner::AddThreadModifier))
        .def("AddConstraint", &ktt::Tuner::AddConstraint)
        .def("SetProfiledDefinitions", &ktt::Tuner::SetProfiledDefinitions)
        .def("AddArgumentVectorChar", py::overload_cast<const std::vector<int8_t>&, const ktt::ArgumentAccessType>(&ktt::Tuner::AddArgumentVector<int8_t>))
        .def("AddArgumentVectorShort", py::overload_cast<const std::vector<int16_t>&, const ktt::ArgumentAccessType>(&ktt::Tuner::AddArgumentVector<int16_t>))
        .def("AddArgumentVectorInt", py::overload_cast<const std::vector<int32_t>&, const ktt::ArgumentAccessType>(&ktt::Tuner::AddArgumentVector<int32_t>))
        .def("AddArgumentVectorLong", py::overload_cast<const std::vector<int64_t>&, const ktt::ArgumentAccessType>(&ktt::Tuner::AddArgumentVector<int64_t>))
        .def("AddArgumentVectorFloat", py::overload_cast<const std::vector<float>&, const ktt::ArgumentAccessType>(&ktt::Tuner::AddArgumentVector<float>))
        .def("AddArgumentVectorDouble", py::overload_cast<const std::vector<double>&, const ktt::ArgumentAccessType>(&ktt::Tuner::AddArgumentVector<double>))
        .def("AddArgumentVectorChar", py::overload_cast<std::vector<int8_t>&, const ktt::ArgumentAccessType, const ktt::ArgumentMemoryLocation,
            const ktt::ArgumentManagementType, const bool>(&ktt::Tuner::AddArgumentVector<int8_t>))
        .def("AddArgumentVectorShort", py::overload_cast<std::vector<int16_t>&, const ktt::ArgumentAccessType, const ktt::ArgumentMemoryLocation,
            const ktt::ArgumentManagementType, const bool>(&ktt::Tuner::AddArgumentVector<int16_t>))
        .def("AddArgumentVectorInt", py::overload_cast<std::vector<int32_t>&, const ktt::ArgumentAccessType, const ktt::ArgumentMemoryLocation,
            const ktt::ArgumentManagementType, const bool>(&ktt::Tuner::AddArgumentVector<int32_t>))
        .def("AddArgumentVectorLong", py::overload_cast<std::vector<int64_t>&, const ktt::ArgumentAccessType, const ktt::ArgumentMemoryLocation,
            const ktt::ArgumentManagementType, const bool>(&ktt::Tuner::AddArgumentVector<int64_t>))
        .def("AddArgumentVectorFloat", py::overload_cast<std::vector<float>&, const ktt::ArgumentAccessType, const ktt::ArgumentMemoryLocation,
            const ktt::ArgumentManagementType, const bool>(&ktt::Tuner::AddArgumentVector<float>))
        .def("AddArgumentVectorDouble", py::overload_cast<std::vector<double>&, const ktt::ArgumentAccessType, const ktt::ArgumentMemoryLocation,
            const ktt::ArgumentManagementType, const bool>(&ktt::Tuner::AddArgumentVector<double>))
        .def("AddArgumentScalarChar", &ktt::Tuner::AddArgumentScalar<int8_t>)
        .def("AddArgumentScalarShort", &ktt::Tuner::AddArgumentScalar<int16_t>)
        .def("AddArgumentScalarInt", &ktt::Tuner::AddArgumentScalar<int32_t>)
        .def("AddArgumentScalarLong", &ktt::Tuner::AddArgumentScalar<int64_t>)
        .def("AddArgumentScalarFloat", &ktt::Tuner::AddArgumentScalar<float>)
        .def("AddArgumentScalarDouble", &ktt::Tuner::AddArgumentScalar<double>)
        .def("AddArgumentLocalChar", &ktt::Tuner::AddArgumentLocal<int8_t>)
        .def("AddArgumentLocalShort", &ktt::Tuner::AddArgumentLocal<int16_t>)
        .def("AddArgumentLocalInt", &ktt::Tuner::AddArgumentLocal<int32_t>)
        .def("AddArgumentLocalLong", &ktt::Tuner::AddArgumentLocal<int64_t>)
        .def("AddArgumentLocalFloat", &ktt::Tuner::AddArgumentLocal<float>)
        .def("AddArgumentLocalDouble", &ktt::Tuner::AddArgumentLocal<double>)
        .def
        (
            "AddArgumentSymbolChar",
            &ktt::Tuner::AddArgumentSymbol<int8_t>,
            py::arg("data"),
            py::arg("symbolName") = std::string()
        )
        .def
        (
            "AddArgumentSymbolShort",
            &ktt::Tuner::AddArgumentSymbol<int16_t>,
            py::arg("data"),
            py::arg("symbolName") = std::string()
        )
        .def
        (
            "AddArgumentSymbolInt",
            &ktt::Tuner::AddArgumentSymbol<int32_t>,
            py::arg("data"),
            py::arg("symbolName") = std::string()
        )
        .def
        (
            "AddArgumentSymbolLong",
            &ktt::Tuner::AddArgumentSymbol<int64_t>,
            py::arg("data"),
            py::arg("symbolName") = std::string()
        )
        .def
        (
            "AddArgumentSymbolFloat",
            &ktt::Tuner::AddArgumentSymbol<float>,
            py::arg("data"),
            py::arg("symbolName") = std::string()
        )
        .def
        (
            "AddArgumentSymbolDouble",
            &ktt::Tuner::AddArgumentSymbol<double>,
            py::arg("data"),
            py::arg("symbolName") = std::string()
        )
        .def("RemoveArgument", &ktt::Tuner::RemoveArgument)
        .def("SetReadOnlyArgumentCache", &ktt::Tuner::SetReadOnlyArgumentCache)
        .def("Run", &ktt::Tuner::Run)
        .def("SetProfiling", &ktt::Tuner::SetProfiling)
        .def("SetProfilingCounters", &ktt::Tuner::SetProfilingCounters)
        .def("SetValidationMethod", &ktt::Tuner::SetValidationMethod)
        .def("SetValidationMode", &ktt::Tuner::SetValidationMode)
        .def("SetValidationRange", &ktt::Tuner::SetValidationRange)
        .def("SetValueComparator", &ktt::Tuner::SetValueComparator)
        .def("SetReferenceComputation", &ktt::Tuner::SetReferenceComputation)
        .def("SetReferenceKernel", &ktt::Tuner::SetReferenceKernel)
        .def("Tune", py::overload_cast<const ktt::KernelId>(&ktt::Tuner::Tune), py::call_guard<py::gil_scoped_release>())
        .def("Tune", py::overload_cast<const ktt::KernelId, std::unique_ptr<ktt::StopCondition>>(&ktt::Tuner::Tune), py::call_guard<py::gil_scoped_release>())
        .def
        (
            "TuneIteration",
            &ktt::Tuner::TuneIteration,
            py::call_guard<py::gil_scoped_release>(),
            py::arg("id"),
            py::arg("output"),
            py::arg("recomputeReference") = false
        )
        .def
        (
            "SimulateKernelTuning",
            &ktt::Tuner::SimulateKernelTuning,
            py::call_guard<py::gil_scoped_release>(),
            py::arg("id"),
            py::arg("results"),
            py::arg("iterations") = 0
        )
        .def("SetSearcher", &ktt::Tuner::SetSearcher)
        .def("ClearData", &ktt::Tuner::ClearData)
        .def("GetBestConfiguration", &ktt::Tuner::GetBestConfiguration)
        .def("CreateConfiguration", &ktt::Tuner::CreateConfiguration)
        .def("GetKernelSource", &ktt::Tuner::GetKernelSource)
        .def("GetKernelDefinitionSource", &ktt::Tuner::GetKernelDefinitionSource)
        .def_static("SetTimeUnit", &ktt::Tuner::SetTimeUnit)
        .def
        (
            "SaveResults",
            &ktt::Tuner::SaveResults,
            py::arg("results"),
            py::arg("filePath"),
            py::arg("format"),
            py::arg("data") = ktt::UserData{}
        )
        .def("LoadResults", [](ktt::Tuner& tuner, const std::string& filePath, const ktt::OutputFormat format) { return tuner.LoadResults(filePath, format); })
        .def
        (
            "LoadResultsWithData",
            [](ktt::Tuner& tuner, const std::string& filePath, const ktt::OutputFormat format)
            {
                ktt::UserData data;
                auto results = tuner.LoadResults(filePath, format, data);
                return std::make_pair(results, data);
            }
        )
        .def("AddComputeQueue", &ktt::Tuner::AddComputeQueue)
        .def("RemoveComputeQueue", &ktt::Tuner::RemoveComputeQueue)
        .def("Synchronize", &ktt::Tuner::Synchronize)
        .def("SetCompilerOptions", &ktt::Tuner::SetCompilerOptions)
        .def("SetGlobalSizeType", &ktt::Tuner::SetGlobalSizeType)
        .def("SetAutomaticGlobalSizeCorrection", &ktt::Tuner::SetAutomaticGlobalSizeCorrection)
        .def("SetKernelCacheCapacity", &ktt::Tuner::SetKernelCacheCapacity)
        .def("GetPlatformInfo", &ktt::Tuner::GetPlatformInfo)
        .def("GetDeviceInfo", &ktt::Tuner::GetDeviceInfo)
        .def("GetCurrentDeviceInfo", &ktt::Tuner::GetCurrentDeviceInfo)
        .def_static("SetLoggingLevel", &ktt::Tuner::SetLoggingLevel)
        .def_static("SetLoggingTarget", py::overload_cast<std::ostream&>(&ktt::Tuner::SetLoggingTarget))
        .def_static("SetLoggingTarget", py::overload_cast<const std::string&>(&ktt::Tuner::SetLoggingTarget));
}

#endif // KTT_PYTHON
