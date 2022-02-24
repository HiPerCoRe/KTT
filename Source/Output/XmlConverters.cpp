#include <Output/TimeConfiguration/TimeConfiguration.h>
#include <Output/XmlConverters.h>
#include <Utility/ErrorHandling/Assert.h>

namespace ktt
{
    
static const int xmlFloatingPointPrecision = 6;

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

std::string GlobalSizeTypeToString(const GlobalSizeType sizeType)
{
    switch (sizeType)
    {
    case GlobalSizeType::OpenCL:
        return "OpenCL";
    case GlobalSizeType::CUDA:
        return "CUDA";
    case GlobalSizeType::Vulkan:
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
    case ResultStatus::CompilationFailed:
        return "CompilationFailed";
    case ResultStatus::DeviceLimitsExceeded:
        return "DeviceLimitsExceeded";
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

ComputeApi ComputeApiFromString(const std::string& string)
{
    if (string == "OpenCL")
    {
        return ComputeApi::OpenCL;
    }
    else if (string == "CUDA")
    {
        return ComputeApi::CUDA;
    }
    else if (string == "Vulkan")
    {
        return ComputeApi::Vulkan;
    }

    KttError("Invalid string value");
    return ComputeApi::OpenCL;
}

GlobalSizeType GlobalSizeTypeFromString(const std::string& string)
{
    if (string == "OpenCL")
    {
        return GlobalSizeType::OpenCL;
    }
    else if (string == "CUDA")
    {
        return GlobalSizeType::CUDA;
    }
    else if (string == "Vulkan")
    {
        return GlobalSizeType::Vulkan;
    }

    KttError("Invalid string value");
    return GlobalSizeType::OpenCL;
}

TimeUnit TimeUnitFromString(const std::string& string)
{
    if (string == "Nanoseconds")
    {
        return TimeUnit::Nanoseconds;
    }
    else if (string == "Microseconds")
    {
        return TimeUnit::Microseconds;
    }
    else if (string == "Milliseconds")
    {
        return TimeUnit::Milliseconds;
    }
    else if (string == "Seconds")
    {
        return TimeUnit::Seconds;
    }

    KttError("Invalid string value");
    return TimeUnit::Nanoseconds;
}

ResultStatus ResultStatusFromString(const std::string& string)
{
    if (string == "Ok")
    {
        return ResultStatus::Ok;
    }
    else if (string == "ComputationFailed")
    {
        return ResultStatus::ComputationFailed;
    }
    else if (string == "ValidationFailed")
    {
        return ResultStatus::ValidationFailed;
    }
    else if (string == "CompilationFailed")
    {
        return ResultStatus::CompilationFailed;
    }
    else if (string == "DeviceLimitsExceeded")
    {
        return ResultStatus::DeviceLimitsExceeded;
    }

    KttError("Invalid string value");
    return ResultStatus::Ok;
}

ProfilingCounterType ProfilingCounterTypeFromString(const std::string& string)
{
    if (string == "Int")
    {
        return ProfilingCounterType::Int;
    }
    else if (string == "UnsignedInt")
    {
        return ProfilingCounterType::UnsignedInt;
    }
    else if (string == "Double")
    {
        return ProfilingCounterType::Double;
    }
    else if (string == "Percent")
    {
        return ProfilingCounterType::Percent;
    }
    else if (string == "Throughput")
    {
        return ProfilingCounterType::Throughput;
    }
    else if (string == "UtilizationLevel")
    {
        return ProfilingCounterType::UtilizationLevel;
    }

    KttError("Invalid string value");
    return ProfilingCounterType::Int;
}

void AppendMetadata(pugi::xml_node parent, const TunerMetadata& metadata)
{
    pugi::xml_node node = parent.append_child("Metadata");
    node.append_attribute("ComputeApi").set_value(ComputeApiToString(metadata.GetComputeApi()).c_str());
    node.append_attribute("GlobalSizeType").set_value(GlobalSizeTypeToString(metadata.GetGlobalSizeType()).c_str());
    node.append_attribute("Platform").set_value(metadata.GetPlatformName().c_str());
    node.append_attribute("Device").set_value(metadata.GetDeviceName().c_str());
    node.append_attribute("KttVersion").set_value(metadata.GetKttVersion().c_str());
    node.append_attribute("Timestamp").set_value(metadata.GetTimestamp().c_str());
    node.append_attribute("TimeUnit").set_value(TimeUnitToString(metadata.GetTimeUnit()).c_str());
}

TunerMetadata ParseMetadata(const pugi::xml_node node)
{
    TunerMetadata metadata;

    metadata.SetComputeApi(ComputeApiFromString(node.attribute("ComputeApi").value()));
    metadata.SetGlobalSizeType(GlobalSizeTypeFromString(node.attribute("GlobalSizeType").value()));
    metadata.SetPlatformName(node.attribute("Platform").value());
    metadata.SetDeviceName(node.attribute("Device").value());
    metadata.SetKttVersion(node.attribute("KttVersion").value());
    metadata.SetTimestamp(node.attribute("Timestamp").value());
    metadata.SetTimeUnit(TimeUnitFromString(node.attribute("TimeUnit").value()));

    return metadata;
}

void AppendUserData(pugi::xml_node parent, const UserData& data)
{
    pugi::xml_node node = parent.append_child("UserData");

    for (const auto& pair : data)
    {
        pugi::xml_node pairNode = node.append_child("Pair");
        pairNode.append_attribute("Key").set_value(pair.first.c_str());
        pairNode.append_attribute("Value").set_value(pair.second.c_str());
    }
}

UserData ParseUserData(const pugi::xml_node node)
{
    UserData data;

    for (const auto entry : node.children())
    {
        data[entry.attribute("Key").value()] = entry.attribute("Value").value();
    }

    return data;
}

void AppendPair(pugi::xml_node parent, const ParameterPair& pair)
{
    pugi::xml_node node = parent.append_child("Pair");
    node.append_attribute("IsDouble").set_value(pair.HasValueDouble());
    node.append_attribute("Name").set_value(pair.GetName().c_str());
    pugi::xml_attribute value = node.append_attribute("Value");

    if (pair.HasValueDouble())
    {
        value.set_value(pair.GetValueDouble(), xmlFloatingPointPrecision);
    }
    else
    {
        value.set_value(pair.GetValue());
    }
}

ParameterPair ParsePair(const pugi::xml_node node)
{
    const std::string name = node.attribute("Name").value();
    const bool isDouble = node.attribute("IsDouble").as_bool();
    ParameterPair pair;

    if (isDouble)
    {
        const double value = node.attribute("Value").as_double();
        pair = ParameterPair(name, value);
    }
    else
    {
        const uint64_t value = node.attribute("Value").as_ullong();
        pair = ParameterPair(name, value);
    }

    return pair;
}

void AppendVector(pugi::xml_node parent, const DimensionVector& vector, const std::string& tag)
{
    pugi::xml_node node = parent.append_child(tag.c_str());
    node.append_attribute("X").set_value(vector.GetSizeX());
    node.append_attribute("Y").set_value(vector.GetSizeY());
    node.append_attribute("Z").set_value(vector.GetSizeZ());
}

DimensionVector ParseVector(const pugi::xml_node node, const std::string& tag)
{
    const auto vectorNode = node.child(tag.c_str());
    const size_t x = vectorNode.attribute("X").as_ullong();
    const size_t y = vectorNode.attribute("Y").as_ullong();
    const size_t z = vectorNode.attribute("Z").as_ullong(); 
    return DimensionVector(x, y, z);
}

void AppendConfiguration(pugi::xml_node parent, const KernelConfiguration& configuration)
{
    pugi::xml_node node = parent.append_child("Configuration");

    for (const auto& pair : configuration.GetPairs())
    {
        AppendPair(node, pair);
    }
}

KernelConfiguration ParseConfiguration(const pugi::xml_node node)
{
    std::vector<ParameterPair> pairs;

    for (const auto pair : node.children())
    {
        pairs.push_back(ParsePair(pair));
    }

    return KernelConfiguration(pairs);
}

void AppendCounter(pugi::xml_node parent, const KernelProfilingCounter& counter)
{
    pugi::xml_node node = parent.append_child("Counter");
    node.append_attribute("Name").set_value(counter.GetName().c_str());
    node.append_attribute("Type").set_value(ProfilingCounterTypeToString(counter.GetType()).c_str());

    switch (counter.GetType())
    {
    case ProfilingCounterType::Int:
        node.append_attribute("Value").set_value(counter.GetValueInt());
        break;
    case ProfilingCounterType::UnsignedInt:
    case ProfilingCounterType::Throughput:
    case ProfilingCounterType::UtilizationLevel:
        node.append_attribute("Value").set_value(counter.GetValueUint());
        break;
    case ProfilingCounterType::Double:
    case ProfilingCounterType::Percent:
        node.append_attribute("Value").set_value(counter.GetValueDouble(), xmlFloatingPointPrecision);
        break;
    default:
        KttError("Unhandled profiling counter type value");
    }
}

KernelProfilingCounter ParseCounter(const pugi::xml_node node)
{
    const std::string name = node.attribute("Name").value();
    const ProfilingCounterType type = ProfilingCounterTypeFromString(node.attribute("Type").value());

    KernelProfilingCounter counter;
    const pugi::xml_attribute value = node.attribute("Value");

    switch (type)
    {
    case ProfilingCounterType::Int:
    {
        const int64_t valueInt = value.as_llong();
        counter = KernelProfilingCounter(name, type, valueInt);
        break;
    }
    case ProfilingCounterType::UnsignedInt:
    case ProfilingCounterType::Throughput:
    case ProfilingCounterType::UtilizationLevel:
    {
        const uint64_t valueUint = value.as_ullong();
        counter = KernelProfilingCounter(name, type, valueUint);
        break;
    }
    case ProfilingCounterType::Double:
    case ProfilingCounterType::Percent:
    {
        const double valueDouble = value.as_double();
        counter = KernelProfilingCounter(name, type, valueDouble);
        break;
    }
    default:
        KttError("Unhandled profiling counter type value");
    }

    return counter;
}

void AppendCompilationData(pugi::xml_node parent, const KernelCompilationData& data)
{
    pugi::xml_node node = parent.append_child("CompilationData");
    node.append_attribute("MaxWorkGroupSize").set_value(data.m_MaxWorkGroupSize);
    node.append_attribute("LocalMemorySize").set_value(data.m_LocalMemorySize);
    node.append_attribute("PrivateMemorySize").set_value(data.m_PrivateMemorySize);
    node.append_attribute("ConstantMemorySize").set_value(data.m_ConstantMemorySize);
    node.append_attribute("RegistersCount").set_value(data.m_RegistersCount);
}

KernelCompilationData ParseCompilationData(const pugi::xml_node node)
{
    KernelCompilationData data;

    data.m_MaxWorkGroupSize = node.attribute("MaxWorkGroupSize").as_ullong();
    data.m_LocalMemorySize = node.attribute("LocalMemorySize").as_ullong();
    data.m_PrivateMemorySize = node.attribute("PrivateMemorySize").as_ullong();
    data.m_ConstantMemorySize = node.attribute("ConstantMemorySize").as_ullong();
    data.m_RegistersCount = node.attribute("RegistersCount").as_ullong();

    return data;
}

void AppendProfilingData(pugi::xml_node parent, const KernelProfilingData& data)
{
    pugi::xml_node node = parent.append_child("ProfilingData");
    node.append_attribute("RemainingProfilingRuns").set_value(data.GetRemainingProfilingRuns());

    pugi::xml_node counters = node.append_child("Counters");

    for (const auto& counter : data.GetCounters())
    {
        AppendCounter(counters, counter);
    }
}

KernelProfilingData ParseProfilingData(const pugi::xml_node node)
{
    const uint64_t remainingRuns = node.attribute("RemainingProfilingRuns").as_ullong();

    if (remainingRuns == 0)
    {
        std::vector<KernelProfilingCounter> counters;

        for (const auto counter : node.child("Counters").children())
        {
            counters.push_back(ParseCounter(counter));
        }

        return KernelProfilingData(counters);
    }

    return KernelProfilingData(remainingRuns);
}

void AppendComputationResult(pugi::xml_node parent, const ComputationResult& result)
{
    const auto& time = TimeConfiguration::GetInstance();

    pugi::xml_node node = parent.append_child("ComputationResult");
    node.append_attribute("KernelFunction").set_value(result.GetKernelFunction().c_str());
    node.append_attribute("Duration").set_value(time.ConvertFromNanosecondsDouble(result.GetDuration()), xmlFloatingPointPrecision);
    node.append_attribute("Overhead").set_value(time.ConvertFromNanosecondsDouble(result.GetOverhead()), xmlFloatingPointPrecision);
    AppendVector(node, result.GetGlobalSize(), "GlobalSize");
    AppendVector(node, result.GetLocalSize(), "LocalSize");

    if (result.HasCompilationData())
    {
        AppendCompilationData(node, result.GetCompilationData());
    }

    if (result.HasProfilingData())
    {
        AppendProfilingData(node, result.GetProfilingData());
    }

    if (result.HasPowerData())
    {
        node.append_attribute("PowerUsage").set_value(result.GetPowerUsage());
        node.append_attribute("EnergyConsumption").set_value(result.GetEnergyConsumption(), xmlFloatingPointPrecision);
    }
}

ComputationResult ParseComputationResult(const pugi::xml_node node)
{
    const std::string kernelFunction = node.attribute("KernelFunction").value();
    ComputationResult result(kernelFunction);

    const auto& time = TimeConfiguration::GetInstance();
    const double duration = node.attribute("Duration").as_double();
    const Nanoseconds durationNs = time.ConvertToNanosecondsDouble(duration);
    const double overhead = node.attribute("Overhead").as_double();
    const Nanoseconds overheadNs = time.ConvertToNanosecondsDouble(overhead);
    result.SetDurationData(durationNs, overheadNs);

    const DimensionVector globalSize = ParseVector(node, "GlobalSize");
    const DimensionVector localSize = ParseVector(node, "LocalSize");
    result.SetSizeData(globalSize, localSize);

    const auto compilationDataNode = node.child("CompilationData");

    if (!compilationDataNode.empty())
    {
        KernelCompilationData data = ParseCompilationData(compilationDataNode);
        auto uniqueData = std::make_unique<KernelCompilationData>(data);
        result.SetCompilationData(std::move(uniqueData));
    }

    const auto profilingDataNode = node.child("ProfilingData");

    if (!profilingDataNode.empty())
    {
        KernelProfilingData data = ParseProfilingData(profilingDataNode);
        auto uniqueData = std::make_unique<KernelProfilingData>(data);
        result.SetProfilingData(std::move(uniqueData));
    }

    const auto powerUsage = node.attribute("PowerUsage");

    if (!powerUsage.empty())
    {
        const uint32_t powerUsageValue = powerUsage.as_uint();
        result.SetPowerUsage(powerUsageValue);
    }

    return result;
}

void AppendKernelResult(pugi::xml_node parent, const KernelResult& result)
{
    const auto& time = TimeConfiguration::GetInstance();

    pugi::xml_node node = parent.append_child("KernelResult");
    node.append_attribute("KernelName").set_value(result.GetKernelName().c_str());
    node.append_attribute("Status").set_value(ResultStatusToString(result.GetStatus()).c_str());
    node.append_attribute("TotalDuration").set_value(time.ConvertFromNanosecondsDouble(result.GetTotalDuration()),
        xmlFloatingPointPrecision);
    node.append_attribute("TotalOverhead").set_value(time.ConvertFromNanosecondsDouble(result.GetTotalOverhead()),
        xmlFloatingPointPrecision);
    node.append_attribute("ExtraDuration").set_value(time.ConvertFromNanosecondsDouble(result.GetExtraDuration()),
        xmlFloatingPointPrecision);
    node.append_attribute("ExtraOverhead").set_value(time.ConvertFromNanosecondsDouble(result.GetExtraOverhead()),
        xmlFloatingPointPrecision);
    AppendConfiguration(node, result.GetConfiguration());

    pugi::xml_node computationResults = node.append_child("ComputationResults");

    for (const auto& computationResult : result.GetResults())
    {
        AppendComputationResult(computationResults, computationResult);
    }
}

KernelResult ParseKernelResult(const pugi::xml_node node)
{
    const std::string kernelName = node.attribute("KernelName").value();
    const KernelConfiguration configuration = ParseConfiguration(node.child("Configuration"));

    std::vector<ComputationResult> results;

    for (const auto computationResult : node.child("ComputationResults").children())
    {
        results.push_back(ParseComputationResult(computationResult));
    }

    KernelResult result(kernelName, configuration, results);

    const ResultStatus status = ResultStatusFromString(node.attribute("Status").value());
    result.SetStatus(status);

    const auto& time = TimeConfiguration::GetInstance();

    const double extraDuration = node.attribute("ExtraDuration").as_double();
    const Nanoseconds extraDurationNs = time.ConvertToNanosecondsDouble(extraDuration);
    result.SetExtraDuration(extraDurationNs);

    const double extraOverhead = node.attribute("ExtraOverhead").as_double();
    const Nanoseconds extraOverheadNs = time.ConvertToNanosecondsDouble(extraOverhead);
    result.SetExtraOverhead(extraOverheadNs);

    return result;
}

} // namespace ktt
