#ifdef KTT_PYTHON

#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Ktt.h>

namespace py = pybind11;

void InitializePythonDataHolders(py::module_& module)
{
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
        .def("GetName", &ktt::ParameterPair::GetName, py::return_value_policy::reference)
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
        .def("GetPairs", &ktt::KernelConfiguration::GetPairs, py::return_value_policy::reference)
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
        .def("GetName", &ktt::DeviceInfo::GetName, py::return_value_policy::reference)
        .def("GetVendor", &ktt::DeviceInfo::GetVendor, py::return_value_policy::reference)
        .def("GetExtensions", &ktt::DeviceInfo::GetExtensions, py::return_value_policy::reference)
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
        .def("GetName", &ktt::PlatformInfo::GetName, py::return_value_policy::reference)
        .def("GetVendor", &ktt::PlatformInfo::GetVendor, py::return_value_policy::reference)
        .def("GetVersion", &ktt::PlatformInfo::GetVersion, py::return_value_policy::reference)
        .def("GetExtensions", &ktt::PlatformInfo::GetExtensions, py::return_value_policy::reference)
        .def("GetString", &ktt::PlatformInfo::GetString)
        .def("SetVendor", &ktt::PlatformInfo::SetVendor)
        .def("SetVersion", &ktt::PlatformInfo::SetVersion)
        .def("SetExtensions", &ktt::PlatformInfo::SetExtensions)
        .def("__repr__", &ktt::PlatformInfo::GetString);

    py::class_<ktt::BufferOutputDescriptor>(module, "BufferOutputDescriptor")
        .def
        (
            py::init([](const ktt::ArgumentId id, py::buffer buffer)
            {
                void* outputDestination = buffer.request(true).ptr;
                return ktt::BufferOutputDescriptor(id, outputDestination);
            })
        )
        .def
        (
            py::init([](const ktt::ArgumentId id, py::buffer buffer, const size_t size)
            {
                void* outputDestination = buffer.request(true).ptr;
                return ktt::BufferOutputDescriptor(id, outputDestination, size);
            })
        )
        .def("GetArgumentId", &ktt::BufferOutputDescriptor::GetArgumentId)
        .def("GetOutputDestination", &ktt::BufferOutputDescriptor::GetOutputDestination)
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
        .def("GetName", &ktt::KernelProfilingCounter::GetName, py::return_value_policy::reference)
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
        .def("GetCounter", &ktt::KernelProfilingData::GetCounter, py::return_value_policy::reference)
        .def("GetCounters", &ktt::KernelProfilingData::GetCounters, py::return_value_policy::reference)
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
        .def("SetCompilationData", &ktt::ComputationResult::SetCompilationData)
        .def("SetProfilingData", &ktt::ComputationResult::SetProfilingData)
        .def("GetKernelFunction", &ktt::ComputationResult::GetKernelFunction, py::return_value_policy::reference)
        .def("GetGlobalSize", &ktt::ComputationResult::GetGlobalSize, py::return_value_policy::reference)
        .def("GetLocalSize", &ktt::ComputationResult::GetLocalSize, py::return_value_policy::reference)
        .def("GetDuration", &ktt::ComputationResult::GetDuration)
        .def("GetOverhead", &ktt::ComputationResult::GetOverhead)
        .def("HasCompilationData", &ktt::ComputationResult::HasCompilationData)
        .def("GetCompilationData", &ktt::ComputationResult::GetCompilationData, py::return_value_policy::reference)
        .def("HasProfilingData", &ktt::ComputationResult::HasProfilingData)
        .def("GetProfilingData", &ktt::ComputationResult::GetProfilingData, py::return_value_policy::reference)
        .def("HasRemainingProfilingRuns", &ktt::ComputationResult::HasRemainingProfilingRuns)
        .def("HasPowerData", &ktt::ComputationResult::HasPowerData)
        .def("GetPowerUsage", &ktt::ComputationResult::GetPowerUsage)
        .def("GetEnergyConsumption", &ktt::ComputationResult::GetEnergyConsumption)
        .def("assign", &ktt::ComputationResult::operator=);

    py::class_<ktt::KernelResult>(module, "KernelResult")
        .def(py::init<>())
        .def(py::init<const std::string&, const ktt::KernelConfiguration&>())
        .def(py::init<const std::string&, const ktt::KernelConfiguration&, const std::vector<ktt::ComputationResult>&>())
        .def("SetStatus", &ktt::KernelResult::SetStatus)
        .def("SetExtraDuration", &ktt::KernelResult::SetExtraDuration)
        .def("SetExtraOverhead", &ktt::KernelResult::SetExtraOverhead)
        .def("GetKernelName", &ktt::KernelResult::GetKernelName, py::return_value_policy::reference)
        .def("GetResults", &ktt::KernelResult::GetResults, py::return_value_policy::reference)
        .def("GetConfiguration", &ktt::KernelResult::GetConfiguration, py::return_value_policy::reference)
        .def("GetStatus", &ktt::KernelResult::GetStatus)
        .def("GetKernelDuration", &ktt::KernelResult::GetKernelDuration)
        .def("GetKernelOverhead", &ktt::KernelResult::GetKernelOverhead)
        .def("GetExtraDuration", &ktt::KernelResult::GetExtraDuration)
        .def("GetExtraOverhead", &ktt::KernelResult::GetExtraOverhead)
        .def("GetTotalDuration", &ktt::KernelResult::GetTotalDuration)
        .def("GetTotalOverhead", &ktt::KernelResult::GetTotalOverhead)
        .def("IsValid", &ktt::KernelResult::IsValid)
        .def("HasRemainingProfilingRuns", &ktt::KernelResult::HasRemainingProfilingRuns);
}

#endif // KTT_PYTHON
