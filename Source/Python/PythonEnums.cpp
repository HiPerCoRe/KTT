#ifdef KTT_PYTHON

#include <pybind11/pybind11.h>

#include <Ktt.h>

namespace py = pybind11;

void InitializePythonEnums(py::module_& module)
{
    py::enum_<ktt::ArgumentAccessType>(module, "ArgumentAccessType")
        .value("Undefined", ktt::ArgumentAccessType::Undefined)
        .value("ReadOnly", ktt::ArgumentAccessType::ReadOnly)
        .value("WriteOnly", ktt::ArgumentAccessType::WriteOnly)
        .value("ReadWrite", ktt::ArgumentAccessType::ReadWrite);

    py::enum_<ktt::ArgumentDataType>(module, "ArgumentDataType")
        .value("Char", ktt::ArgumentDataType::Char)
        .value("UnsignedChar", ktt::ArgumentDataType::UnsignedChar)
        .value("Short", ktt::ArgumentDataType::Short)
        .value("UnsignedShort", ktt::ArgumentDataType::UnsignedShort)
        .value("Int", ktt::ArgumentDataType::Int)
        .value("UnsignedInt", ktt::ArgumentDataType::UnsignedInt)
        .value("Long", ktt::ArgumentDataType::Long)
        .value("UnsignedLong", ktt::ArgumentDataType::UnsignedLong)
        .value("Half", ktt::ArgumentDataType::Half)
        .value("Float", ktt::ArgumentDataType::Float)
        .value("Double", ktt::ArgumentDataType::Double)
        .value("Custom", ktt::ArgumentDataType::Custom);

    py::enum_<ktt::ArgumentManagementType>(module, "ArgumentManagementType")
        .value("Framework", ktt::ArgumentManagementType::Framework)
        .value("User", ktt::ArgumentManagementType::User);

    py::enum_<ktt::ArgumentMemoryLocation>(module, "ArgumentMemoryLocation")
        .value("Undefined", ktt::ArgumentMemoryLocation::Undefined)
        .value("Device", ktt::ArgumentMemoryLocation::Device)
        .value("Host", ktt::ArgumentMemoryLocation::Host)
        .value("HostZeroCopy", ktt::ArgumentMemoryLocation::HostZeroCopy)
        .value("Unified", ktt::ArgumentMemoryLocation::Unified);

    py::enum_<ktt::ArgumentMemoryType>(module, "ArgumentMemoryType")
        .value("Scalar", ktt::ArgumentMemoryType::Scalar)
        .value("Vector", ktt::ArgumentMemoryType::Vector)
        .value("Local", ktt::ArgumentMemoryType::Local)
        .value("Symbol", ktt::ArgumentMemoryType::Symbol);

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

    py::enum_<ktt::GlobalSizeType>(module, "GlobalSizeType")
        .value("OpenCL", ktt::GlobalSizeType::OpenCL)
        .value("CUDA", ktt::GlobalSizeType::CUDA)
        .value("Vulkan", ktt::GlobalSizeType::Vulkan);

    py::enum_<ktt::KernelRunMode>(module, "KernelRunMode")
        .value("Running", ktt::KernelRunMode::Running)
        .value("OfflineTuning", ktt::KernelRunMode::OfflineTuning)
        .value("OnlineTuning", ktt::KernelRunMode::OnlineTuning)
        .value("ResultValidation", ktt::KernelRunMode::ResultValidation);

    py::enum_<ktt::LoggingLevel>(module, "LoggingLevel")
        .value("Off", ktt::LoggingLevel::Off)
        .value("Error", ktt::LoggingLevel::Error)
        .value("Warning", ktt::LoggingLevel::Warning)
        .value("Info", ktt::LoggingLevel::Info)
        .value("Debug", ktt::LoggingLevel::Debug);

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

    py::enum_<ktt::ModifierType>(module, "ModifierType")
        .value("Global", ktt::ModifierType::Global)
        .value("Local", ktt::ModifierType::Local);

    py::enum_<ktt::OutputFormat>(module, "OutputFormat")
        .value("JSON", ktt::OutputFormat::JSON)
        .value("XML", ktt::OutputFormat::XML);

    py::enum_<ktt::ParameterValueType>(module, "ParameterValueType")
        .value("Int", ktt::ParameterValueType::Int)
        .value("UnsignedInt", ktt::ParameterValueType::UnsignedInt)
        .value("Double", ktt::ParameterValueType::Double)
        .value("Bool", ktt::ParameterValueType::Bool)
        .value("String", ktt::ParameterValueType::String);

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

    py::enum_<ktt::TimeUnit>(module, "TimeUnit")
        .value("Nanoseconds", ktt::TimeUnit::Nanoseconds)
        .value("Microseconds", ktt::TimeUnit::Microseconds)
        .value("Milliseconds", ktt::TimeUnit::Milliseconds)
        .value("Seconds", ktt::TimeUnit::Seconds);

    py::enum_<ktt::ValidationMethod>(module, "ValidationMethod")
        .value("AbsoluteDifference", ktt::ValidationMethod::AbsoluteDifference)
        .value("SideBySideComparison", ktt::ValidationMethod::SideBySideComparison)
        .value("SideBySideRelativeComparison", ktt::ValidationMethod::SideBySideRelativeComparison);

    py::enum_<ktt::ValidationMode>(module, "ValidationMode", py::arithmetic())
        .value("None", ktt::ValidationMode::None)
        .value("Running", ktt::ValidationMode::Running)
        .value("OfflineTuning", ktt::ValidationMode::OfflineTuning)
        .value("OnlineTuning", ktt::ValidationMode::OnlineTuning)
        .value("All", ktt::ValidationMode::All);
}

#endif // KTT_PYTHON
