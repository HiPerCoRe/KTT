#ifdef KTT_PYTHON

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Ktt.h>

namespace py = pybind11;

void InitializePythonTuner(py::module_& module)
{
    py::class_<ktt::Tuner>(module, "Tuner")
        .def(py::init<const ktt::PlatformIndex, const ktt::DeviceIndex, const ktt::ComputeApi>())
        .def(py::init<const ktt::PlatformIndex, const ktt::DeviceIndex, const ktt::ComputeApi, const uint32_t>())
        .def
        (
            "AddKernelDefinition",
            py::overload_cast<const std::string&, const std::string&, const ktt::DimensionVector&, const ktt::DimensionVector&,
                const std::vector<std::string>&>(&ktt::Tuner::AddKernelDefinition),
            py::arg("name"),
            py::arg("source"),
            py::arg("globalSize"),
            py::arg("localSize"),
            py::arg("typeNames") = std::vector<std::string>{}
        )
        .def
        (
            "AddKernelDefinition",
            py::overload_cast<const std::string&, const std::string&, const std::vector<std::string>&>(&ktt::Tuner::AddKernelDefinition),
            py::arg("name"),
            py::arg("source"),
            py::arg("typeNames") = std::vector<std::string>{}
        )
        .def
        (
            "AddKernelDefinitionFromFile",
            py::overload_cast<const std::string&, const std::string&, const ktt::DimensionVector&, const ktt::DimensionVector&,
            const std::vector<std::string>&>(&ktt::Tuner::AddKernelDefinitionFromFile),
            py::arg("name"),
            py::arg("filePath"),
            py::arg("globalSize"),
            py::arg("localSize"),
            py::arg("typeNames") = std::vector<std::string>{}
        )
        .def
        (
            "AddKernelDefinitionFromFile",
            py::overload_cast<const std::string&, const std::string&, const std::vector<std::string>&>(&ktt::Tuner::AddKernelDefinitionFromFile),
            py::arg("name"),
            py::arg("filePath"),
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
            &ktt::Tuner::AddParameter<uint64_t>,
            py::arg("id"),
            py::arg("name"),
            py::arg("values"),
            py::arg("group") = std::string()
        )
        .def
        (
            "AddParameterInt",
            &ktt::Tuner::AddParameter<int64_t>,
            py::arg("id"),
            py::arg("name"),
            py::arg("values"),
            py::arg("group") = std::string()
        )
        .def
        (
            "AddParameterUint",
            &ktt::Tuner::AddParameter<uint64_t>,
            py::arg("id"),
            py::arg("name"),
            py::arg("values"),
            py::arg("group") = std::string()
        )
        .def
        (
            "AddParameterDouble",
            &ktt::Tuner::AddParameter<double>,
            py::arg("id"),
            py::arg("name"),
            py::arg("values"),
            py::arg("group") = std::string()
        )
        .def
        (
            "AddParameterBool",
            &ktt::Tuner::AddParameter<bool>,
            py::arg("id"),
            py::arg("name"),
            py::arg("values"),
            py::arg("group") = std::string()
        )
        .def
        (
            "AddParameterString",
            &ktt::Tuner::AddParameter<std::string>,
            py::arg("id"),
            py::arg("name"),
            py::arg("values"),
            py::arg("group") = std::string()
        )
        .def("AddThreadModifier", py::overload_cast<const ktt::KernelId, const std::vector<ktt::KernelDefinitionId>&, const ktt::ModifierType,
            const ktt::ModifierDimension, const std::vector<std::string>&, ktt::ModifierFunction>(&ktt::Tuner::AddThreadModifier))
        .def("AddThreadModifier", py::overload_cast<const ktt::KernelId, const std::vector<ktt::KernelDefinitionId>&, const ktt::ModifierType,
            const ktt::ModifierDimension, const std::string&, const ktt::ModifierAction>(&ktt::Tuner::AddThreadModifier))
        .def("AddScriptThreadModifier", &ktt::Tuner::AddScriptThreadModifier)
        .def("AddConstraint", &ktt::Tuner::AddConstraint)
        .def("AddGenericConstraint", &ktt::Tuner::AddGenericConstraint)
        .def("AddScriptConstraint", &ktt::Tuner::AddScriptConstraint)
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
        .def("Run", py::overload_cast<const ktt::KernelId, const ktt::KernelConfiguration&,
            const std::vector<ktt::BufferOutputDescriptor>&>(&ktt::Tuner::Run))
        .def("Run", py::overload_cast<const ktt::KernelId, const ktt::KernelConfiguration&, const ktt::KernelDimensions&,
            const std::vector<ktt::BufferOutputDescriptor>&>(&ktt::Tuner::Run))
        .def("SetProfiling", &ktt::Tuner::SetProfiling)
        .def("SetProfilingCounters", &ktt::Tuner::SetProfilingCounters)
        .def("SetValidationMethod", &ktt::Tuner::SetValidationMethod)
        .def("SetValidationMode", &ktt::Tuner::SetValidationMode)
        .def("SetValidationRange", &ktt::Tuner::SetValidationRange)
        .def("SetValueComparator", &ktt::Tuner::SetValueComparator)
        .def
        (
            "SetReferenceComputation",
            [](ktt::Tuner& tuner, const ktt::ArgumentId id, const size_t referenceSize, std::function<void(py::memoryview)> reference)
            {
                ktt::ReferenceComputation actualReference = [reference, referenceSize](void* buffer)
                {
                    py::gil_scoped_acquire acquire;
                    auto view = py::memoryview::from_memory(buffer, referenceSize);
                    reference(view);
                };

                tuner.SetReferenceComputation(id, actualReference);
            }
        )
        .def
        (
            "SetReferenceKernel",
            &ktt::Tuner::SetReferenceKernel,
            py::arg("id"),
            py::arg("referenceId"),
            py::arg("configuration"),
            py::arg("dimensions") = ktt::KernelDimensions{}
        )
        .def
        (
            "Tune",
            py::overload_cast<const ktt::KernelId, std::unique_ptr<ktt::StopCondition>>(&ktt::Tuner::Tune),
            py::call_guard<py::gil_scoped_release>(),
            py::arg("id"),
            py::arg("stopCondition") = nullptr
        )
        .def
        (
            "Tune",
            py::overload_cast<const ktt::KernelId, const ktt::KernelDimensions&, std::unique_ptr<ktt::StopCondition>>(&ktt::Tuner::Tune),
            py::call_guard<py::gil_scoped_release>(),
            py::arg("id"),
            py::arg("dimensions"),
            py::arg("stopCondition") = nullptr
        )
        .def
        (
            "TuneIteration",
            py::overload_cast<const ktt::KernelId, const std::vector<ktt::BufferOutputDescriptor>&, const bool>(&ktt::Tuner::TuneIteration),
            py::call_guard<py::gil_scoped_release>(),
            py::arg("id"),
            py::arg("output"),
            py::arg("recomputeReference") = false
        )
        .def
        (
            "TuneIteration",
            py::overload_cast<const ktt::KernelId, const ktt::KernelDimensions&, const std::vector<ktt::BufferOutputDescriptor>&,
                const bool>(&ktt::Tuner::TuneIteration),
            py::call_guard<py::gil_scoped_release>(),
            py::arg("id"),
            py::arg("dimensions"),
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
        .def("SetProfileBasedSearcher", &ktt::Tuner::SetProfileBasedSearcher)
        .def("InitializeConfigurationData", &ktt::Tuner::InitializeConfigurationData)
        .def("ClearConfigurationData", &ktt::Tuner::ClearConfigurationData)
        .def("ClearData", &ktt::Tuner::ClearConfigurationData)
        .def("GetConfigurationsCount", &ktt::Tuner::GetConfigurationsCount)
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
        .def("WaitForComputeAction", &ktt::Tuner::WaitForComputeAction)
        .def("WaitForTransferAction", &ktt::Tuner::WaitForTransferAction)
        .def("SynchronizeQueue", &ktt::Tuner::SynchronizeQueue)
        .def("SynchronizeQueues", &ktt::Tuner::SynchronizeQueues)
        .def("SynchronizeDevice", &ktt::Tuner::SynchronizeDevice)
        .def
        (
            "SetCompilerOptions",
            &ktt::Tuner::SetCompilerOptions,
            py::arg("options"),
            py::arg("overrideDefault") = false
        )
        .def("SetGlobalSizeType", &ktt::Tuner::SetGlobalSizeType)
        .def("SetAutomaticGlobalSizeCorrection", &ktt::Tuner::SetAutomaticGlobalSizeCorrection)
        .def("SetKernelCacheCapacity", &ktt::Tuner::SetKernelCacheCapacity)
        .def("GetPlatformInfo", &ktt::Tuner::GetPlatformInfo)
        .def("GetDeviceInfo", &ktt::Tuner::GetDeviceInfo)
        .def("GetCurrentDeviceInfo", &ktt::Tuner::GetCurrentDeviceInfo)
        .def_static("SetLoggingLevel", &ktt::Tuner::SetLoggingLevel)
        .def_static("GetLoggingLevel", &ktt::Tuner::GetLoggingLevel)
        .def_static("SetLoggingTarget", py::overload_cast<std::ostream&>(&ktt::Tuner::SetLoggingTarget))
        .def_static("SetLoggingTarget", py::overload_cast<const std::string&>(&ktt::Tuner::SetLoggingTarget));
}

#endif // KTT_PYTHON
