#include <set>

#include <Output/TimeConfiguration/TimeConfiguration.h>
#include <Output/XmlConverters.h>
#include <Utility/ErrorHandling/Assert.h>

namespace ktt
{
    
std::string ComputeApiToString(const ComputeApi api)
{
    switch (api)
    {
    case ComputeApi::OpenCL:
        return "OpenCL";
    case ComputeApi::CUDA:
        return "CUDA";
    case ComputeApi::Vulkan:
        return "Vulkan";
    default:
        KttError("Unhandled value");
        return "";
    }
}

std::string TimeUnitToString(const TimeUnit unit)
{
    switch (unit)
    {
    case TimeUnit::Nanoseconds:
        return "Nanoseconds";
    case TimeUnit::Microseconds:
        return "Microseconds";
    case TimeUnit::Milliseconds:
        return "Milliseconds";
    case TimeUnit::Seconds:
        return "Seconds";
    default:
        KttError("Unhandled value");
        return "";
    }
}

std::string ResultStatusToString(const ResultStatus status)
{
    switch (status)
    {
    case ResultStatus::Ok:
        return "Ok";
    case ResultStatus::ComputationFailed:
        return "ComputationFailed";
    case ResultStatus::ValidationFailed:
        return "ValidationFailed";
    default:
        KttError("Unhandled value");
        return "";
    }
}

std::string ProfilingCounterTypeToString(const ProfilingCounterType type)
{
    switch (type)
    {
    case ProfilingCounterType::Int:
        return "Int";
    case ProfilingCounterType::UnsignedInt:
        return "UnsignedInt";
    case ProfilingCounterType::Double:
        return "Double";
    case ProfilingCounterType::Percent:
        return "Percent";
    case ProfilingCounterType::Throughput:
        return "Throughput";
    case ProfilingCounterType::UtilizationLevel:
        return "UtilizationLevel";
    default:
        KttError("Unhandled value");
        return "";
    }
}

ComputeApi ComputeApiFromString(const std::string& /*string*/)
{
    return ComputeApi::OpenCL;
}

TimeUnit TimeUnitFromString(const std::string& /*string*/)
{
    return TimeUnit::Nanoseconds;
}

ResultStatus ResultStatusFromString(const std::string& /*string*/)
{
    return ResultStatus::Ok;
}

ProfilingCounterType ProfilingCounterTypeFromString(const std::string& /*string*/)
{
    return ProfilingCounterType::Int;
}

void AppendMetadata(pugi::xml_node parent, const TunerMetadata& metadata)
{
    pugi::xml_node node = parent.append_child("Metadata");
    node.append_attribute("ComputeApi").set_value(ComputeApiToString(metadata.GetComputeApi()).c_str());
    node.append_attribute("Platform").set_value(metadata.GetPlatformName().c_str());
    node.append_attribute("Device").set_value(metadata.GetDeviceName().c_str());
    node.append_attribute("KttVersion").set_value(metadata.GetKttVersion().c_str());
    node.append_attribute("TimeUnit").set_value(TimeUnitToString(metadata.GetTimeUnit()).c_str());
}

TunerMetadata ParseMetadata(const pugi::xml_node /*node*/)
{
    return TunerMetadata();
}

void AppendKernelResult(pugi::xml_node parent, const KernelResult& result)
{
    const auto& time = TimeConfiguration::GetInstance();

    pugi::xml_node node = parent.append_child("KernelResult");
    node.append_attribute("KernelName").set_value(result.GetKernelName().c_str());
    node.append_attribute("Status").set_value(ResultStatusToString(result.GetStatus()).c_str());
    node.append_attribute("TotalDuration").set_value(time.ConvertFromNanosecondsDouble(result.GetTotalDuration()));
    node.append_attribute("TotalOverhead").set_value(time.ConvertFromNanosecondsDouble(result.GetTotalOverhead()));
}

KernelResult ParseKernelResult(const pugi::xml_node /*node*/)
{
    return KernelResult();
}

} // namespace ktt
