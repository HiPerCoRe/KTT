#ifdef KTT_PYTHON

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Ktt.h>

namespace py = pybind11;

class PySearcher : public ktt::Searcher
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

PYBIND11_MODULE(ktt, module)
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

    py::register_exception<ktt::KttException>(module, "KttException", PyExc_Exception);

    py::enum_<ktt::ComputeApi>(module, "ComputeApi")
        .value("OpenCL", ktt::ComputeApi::OpenCL)
        .value("CUDA", ktt::ComputeApi::CUDA)
        .value("Vulkan", ktt::ComputeApi::Vulkan);

    py::enum_<ktt::DeviceType>(module, "DeviceType")
        .value("CPU", ktt::DeviceType::CPU)
        .value("GPU", ktt::DeviceType::GPU)
        .value("Custom", ktt::DeviceType::Custom);

    py::enum_<ktt::ExceptionReason>(module, "ExceptionReason")
        .value("General", ktt::ExceptionReason::General)
        .value("CompilerError", ktt::ExceptionReason::CompilerError)
        .value("DeviceLimitsExceeded", ktt::ExceptionReason::DeviceLimitsExceeded);

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

    py::enum_<ktt::ProfilingCounterType>(module, "ProfilingCounterType")
        .value("Int", ktt::ProfilingCounterType::Int)
        .value("UnsignedInt", ktt::ProfilingCounterType::UnsignedInt)
        .value("Double", ktt::ProfilingCounterType::Double)
        .value("Percent", ktt::ProfilingCounterType::Percent)
        .value("Throughput", ktt::ProfilingCounterType::Throughput)
        .value("UtilizationLevel", ktt::ProfilingCounterType::UtilizationLevel);

    py::enum_<ktt::ResultStatus>(module, "ResultStatus")
        .value("Ok", ktt::ResultStatus::Ok)
        .value("ComputationFailed", ktt::ResultStatus::ComputationFailed)
        .value("ValidationFailed", ktt::ResultStatus::ValidationFailed)
        .value("CompilationFailed", ktt::ResultStatus::CompilationFailed)
        .value("DeviceLimitsExceeded", ktt::ResultStatus::DeviceLimitsExceeded);

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

    py::class_<ktt::ParameterPair>(module, "ParameterPair")
        .def(py::init<>())
        .def(py::init<const std::string&, const uint64_t>())
        .def(py::init<const std::string&, const double>())
        .def("SetValue", py::overload_cast<const uint64_t>(&ktt::ParameterPair::SetValue))
        .def("SetValue", py::overload_cast<const double>(&ktt::ParameterPair::SetValue))
        .def("GetName", &ktt::ParameterPair::GetName)
        .def("GetString", &ktt::ParameterPair::GetString)
        .def("GetValueString", &ktt::ParameterPair::GetValueString)
        .def("GetValue", &ktt::ParameterPair::GetValue)
        .def("GetValueDouble", &ktt::ParameterPair::GetValueDouble)
        .def("HasValueDouble", &ktt::ParameterPair::HasValueDouble)
        .def("HasSameValue", &ktt::ParameterPair::HasSameValue)
        .def_static("GetParameterValue", &ktt::ParameterPair::GetParameterValue<uint64_t>)
        .def_static("GetParameterValueDouble", &ktt::ParameterPair::GetParameterValue<double>)
        .def_static("GetParameterValues", &ktt::ParameterPair::GetParameterValues<uint64_t>)
        .def_static("GetParameterValuesDouble", &ktt::ParameterPair::GetParameterValues<double>)
        .def("__repr__", &ktt::ParameterPair::GetString);

    py::class_<ktt::KernelConfiguration>(module, "KernelConfiguration")
        .def(py::init<>())
        .def(py::init<const std::vector<ktt::ParameterPair>&>())
        .def("GetPairs", &ktt::KernelConfiguration::GetPairs)
        .def("IsValid", &ktt::KernelConfiguration::IsValid)
        .def("GeneratePrefix", &ktt::KernelConfiguration::GeneratePrefix)
        .def("GetString", &ktt::KernelConfiguration::GetString)
        .def("Merge", &ktt::KernelConfiguration::Merge)
        .def("GenerateNeighbours", &ktt::KernelConfiguration::GenerateNeighbours)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("__repr__", &ktt::KernelConfiguration::GetString);

    py::class_<ktt::DeviceInfo>(module, "DeviceInfo")
        .def(py::init<const ktt::DeviceIndex, const std::string&>())
        .def("GetIndex", &ktt::DeviceInfo::GetIndex)
        .def("GetName", &ktt::DeviceInfo::GetName)
        .def("GetVendor", &ktt::DeviceInfo::GetVendor)
        .def("GetExtensions", &ktt::DeviceInfo::GetExtensions)
        .def("GetDeviceType", &ktt::DeviceInfo::GetDeviceType)
        .def("GetDeviceTypeString", &ktt::DeviceInfo::GetDeviceTypeString)
        .def("GetGlobalMemorySize", &ktt::DeviceInfo::GetGlobalMemorySize)
        .def("GetLocalMemorySize", &ktt::DeviceInfo::GetLocalMemorySize)
        .def("GetMaxConstantBufferSize", &ktt::DeviceInfo::GetMaxConstantBufferSize)
        .def("GetMaxWorkGroupSize", &ktt::DeviceInfo::GetMaxWorkGroupSize)
        .def("GetMaxComputeUnits", &ktt::DeviceInfo::GetMaxComputeUnits)
        .def("GetString", &ktt::DeviceInfo::GetString)
        .def("SetVendor", &ktt::DeviceInfo::SetVendor)
        .def("SetExtensions", &ktt::DeviceInfo::SetExtensions)
        .def("SetDeviceType", &ktt::DeviceInfo::SetDeviceType)
        .def("SetGlobalMemorySize", &ktt::DeviceInfo::SetGlobalMemorySize)
        .def("SetLocalMemorySize", &ktt::DeviceInfo::SetLocalMemorySize)
        .def("SetMaxConstantBufferSize", &ktt::DeviceInfo::SetMaxConstantBufferSize)
        .def("SetMaxWorkGroupSize", &ktt::DeviceInfo::SetMaxWorkGroupSize)
        .def("SetMaxComputeUnits", &ktt::DeviceInfo::SetMaxComputeUnits)
        .def("__repr__", &ktt::DeviceInfo::GetString);

    py::class_<ktt::PlatformInfo>(module, "PlatformInfo")
        .def(py::init<const ktt::PlatformIndex, const std::string&>())
        .def("GetIndex", &ktt::PlatformInfo::GetIndex)
        .def("GetName", &ktt::PlatformInfo::GetName)
        .def("GetVendor", &ktt::PlatformInfo::GetVendor)
        .def("GetVersion", &ktt::PlatformInfo::GetVersion)
        .def("GetExtensions", &ktt::PlatformInfo::GetExtensions)
        .def("GetString", &ktt::PlatformInfo::GetString)
        .def("SetVendor", &ktt::PlatformInfo::SetVendor)
        .def("SetVersion", &ktt::PlatformInfo::SetVersion)
        .def("SetExtensions", &ktt::PlatformInfo::SetExtensions)
        .def("__repr__", &ktt::PlatformInfo::GetString);

    py::class_<ktt::BufferOutputDescriptor>(module, "BufferOutputDescriptor")
        .def(py::init<const ktt::ArgumentId, void*>())
        .def(py::init<const ktt::ArgumentId, void*, const size_t>())
        .def("GetArgumentId", &ktt::BufferOutputDescriptor::GetArgumentId)
        .def("GetOutputDestination", &ktt::BufferOutputDescriptor::GetOutputDestination, py::return_value_policy::reference)
        .def("GetOutputSize", &ktt::BufferOutputDescriptor::GetOutputSize);

    py::class_<ktt::KernelCompilationData>(module, "KernelCompilationData")
        .def(py::init<>())
        .def_readwrite("m_MaxWorkGroupSize", &ktt::KernelCompilationData::m_MaxWorkGroupSize)
        .def_readwrite("m_LocalMemorySize", &ktt::KernelCompilationData::m_LocalMemorySize)
        .def_readwrite("m_PrivateMemorySize", &ktt::KernelCompilationData::m_PrivateMemorySize)
        .def_readwrite("m_ConstantMemorySize", &ktt::KernelCompilationData::m_ConstantMemorySize)
        .def_readwrite("m_RegistersCount", &ktt::KernelCompilationData::m_RegistersCount);

    py::class_<ktt::KernelProfilingCounter>(module, "KernelProfilingCounter")
        .def(py::init<>())
        .def(py::init<const std::string&, const ktt::ProfilingCounterType, const int64_t>())
        .def(py::init<const std::string&, const ktt::ProfilingCounterType, const uint64_t>())
        .def(py::init<const std::string&, const ktt::ProfilingCounterType, const double>())
        .def("GetName", &ktt::KernelProfilingCounter::GetName)
        .def("GetType", &ktt::KernelProfilingCounter::GetType)
        .def("GetValueInt", &ktt::KernelProfilingCounter::GetValueInt)
        .def("GetValueUint", &ktt::KernelProfilingCounter::GetValueUint)
        .def("GetValueDouble", &ktt::KernelProfilingCounter::GetValueDouble)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self);

    py::class_<ktt::KernelProfilingData>(module, "KernelProfilingData")
        .def(py::init<>())
        .def(py::init<const uint64_t>())
        .def(py::init<const std::vector<ktt::KernelProfilingCounter>&>())
        .def("IsValid", &ktt::KernelProfilingData::IsValid)
        .def("HasCounter", &ktt::KernelProfilingData::HasCounter)
        .def("GetCounter", &ktt::KernelProfilingData::GetCounter)
        .def("GetCounters", &ktt::KernelProfilingData::GetCounters)
        .def("SetCounters", &ktt::KernelProfilingData::SetCounters)
        .def("AddCounter", &ktt::KernelProfilingData::AddCounter)
        .def("HasRemainingProfilingRuns", &ktt::KernelProfilingData::HasRemainingProfilingRuns)
        .def("GetRemainingProfilingRuns", &ktt::KernelProfilingData::GetRemainingProfilingRuns)
        .def("DecreaseRemainingProfilingRuns", &ktt::KernelProfilingData::DecreaseRemainingProfilingRuns);

    py::class_<ktt::ComputationResult>(module, "ComputationResult")
        .def(py::init<>())
        .def(py::init<const std::string&>())
        .def(py::init<const ktt::ComputationResult&>())
        .def("SetDurationData", &ktt::ComputationResult::SetDurationData)
        .def("SetSizeData", &ktt::ComputationResult::SetSizeData)
        // Todo: check pybind11 smart_holder branch for unique_ptr argument passing support
        //.def("SetCompilationData", &ktt::ComputationResult::SetCompilationData)
        //.def("SetProfilingData", &ktt::ComputationResult::SetProfilingData)
        .def("GetKernelFunction", &ktt::ComputationResult::GetKernelFunction)
        .def("GetGlobalSize", &ktt::ComputationResult::GetGlobalSize)
        .def("GetLocalSize", &ktt::ComputationResult::GetLocalSize)
        .def("GetDuration", &ktt::ComputationResult::GetDuration)
        .def("GetOverhead", &ktt::ComputationResult::GetOverhead)
        .def("HasCompilationData", &ktt::ComputationResult::HasCompilationData)
        .def("GetCompilationData", &ktt::ComputationResult::GetCompilationData)
        .def("HasProfilingData", &ktt::ComputationResult::HasProfilingData)
        .def("GetProfilingData", &ktt::ComputationResult::GetProfilingData)
        .def("HasRemainingProfilingRuns", &ktt::ComputationResult::HasRemainingProfilingRuns)
        .def("assign", &ktt::ComputationResult::operator=);

    py::class_<ktt::KernelResult>(module, "KernelResult")
        .def(py::init<>())
        .def(py::init<const std::string&, const ktt::KernelConfiguration&>())
        .def(py::init<const std::string&, const ktt::KernelConfiguration&, const std::vector<ktt::ComputationResult>&>())
        .def("SetStatus", &ktt::KernelResult::SetStatus)
        .def("SetExtraDuration", &ktt::KernelResult::SetExtraDuration)
        .def("SetExtraOverhead", &ktt::KernelResult::SetExtraOverhead)
        .def("GetKernelName", &ktt::KernelResult::GetKernelName)
        .def("GetConfiguration", &ktt::KernelResult::GetConfiguration)
        .def("GetStatus", &ktt::KernelResult::GetStatus)
        .def("GetKernelDuration", &ktt::KernelResult::GetKernelDuration)
        .def("GetKernelOverhead", &ktt::KernelResult::GetKernelOverhead)
        .def("GetExtraDuration", &ktt::KernelResult::GetExtraDuration)
        .def("GetExtraOverhead", &ktt::KernelResult::GetExtraOverhead)
        .def("GetTotalDuration", &ktt::KernelResult::GetTotalDuration)
        .def("GetTotalOverhead", &ktt::KernelResult::GetTotalOverhead)
        .def("IsValid", &ktt::KernelResult::IsValid)
        .def("HasRemainingProfilingRuns", &ktt::KernelResult::HasRemainingProfilingRuns);

    py::class_<ktt::Searcher, PySearcher>(module, "Searcher")
        .def(py::init<>())
        .def("OnInitialize", &ktt::Searcher::OnInitialize)
        .def("OnReset", &ktt::Searcher::OnReset)
        .def("CalculateNextConfiguration", &ktt::Searcher::CalculateNextConfiguration)
        .def("GetCurrentConfiguration", &ktt::Searcher::GetCurrentConfiguration)
        .def("GetIndex", &ktt::Searcher::GetIndex)
        .def("GetRandomConfiguration", &ktt::Searcher::GetRandomConfiguration)
        .def("GetNeighbourConfigurations", &ktt::Searcher::GetNeighbourConfigurations)
        .def("GetConfigurationsCount", &ktt::Searcher::GetConfigurationsCount)
        .def("GetExploredIndices", &ktt::Searcher::GetExploredIndices)
        .def("IsInitialized", &ktt::Searcher::IsInitialized);

    py::class_<ktt::DeterministicSearcher, ktt::Searcher>(module, "DeterministicSearcher")
        .def(py::init<>());

    py::class_<ktt::McmcSearcher, ktt::Searcher>(module, "McmcSearcher")
        .def(py::init<const std::vector<double>&>());

    py::class_<ktt::RandomSearcher, ktt::Searcher>(module, "RandomSearcher")
        .def(py::init<>());

    py::class_<ktt::Tuner>(module, "Tuner")
        .def(py::init<const ktt::PlatformIndex, const ktt::DeviceIndex, const ktt::ComputeApi>())
        .def("RemoveKernelDefinition", &ktt::Tuner::RemoveKernelDefinition);
}

#endif // KTT_PYTHON
