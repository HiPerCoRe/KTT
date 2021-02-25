#include <Output/TimeConfiguration/TimeConfiguration.h>
#include <Output/JsonConverters.h>
#include <Utility/ErrorHandling/Assert.h>

namespace ktt
{
    
void to_json(json& j, const TunerMetadata& metadata)
{
    j = json
    {
        {"ComputeApi", metadata.GetComputeApi()},
        {"Platform", metadata.GetPlatformName()},
        {"Device", metadata.GetDeviceName()},
        {"KttVersion", metadata.GetKttVersion()},
        {"TimeUnit", metadata.GetTimeUnit()},
    };
}

void from_json(const json& j, TunerMetadata& metadata)
{
    metadata.SetComputeApi(j.at("ComputeApi").get<ComputeApi>());
    metadata.SetPlatformName(j.at("Platform").get<std::string>());
    metadata.SetDeviceName(j.at("Device").get<std::string>());
    metadata.SetKttVersion(j.at("KttVersion").get<std::string>());
    metadata.SetTimeUnit(j.at("TimeUnit").get<TimeUnit>());
}

void to_json(json& j, const ParameterPair& pair)
{
    j = json
    {
        {"Name", pair.GetName()},
        {"IsDouble", pair.HasValueDouble()}
    };

    if (pair.HasValueDouble())
    {
        j["Value"] = pair.GetValueDouble();
    }
    else
    {
        j["Value"] = pair.GetValue();
    }
}

void from_json(const json& j, ParameterPair& pair)
{
    static std::vector<std::string> parameterNames;

    std::string name;
    j.at("Name").get_to(name);
    parameterNames.push_back(name);

    bool isDouble;
    j.at("IsDouble").get_to(isDouble);

    if (isDouble)
    {
        double value;
        j.at("Value").get_to(value);
        pair = ParameterPair(parameterNames.back(), value);
    }
    else
    {
        uint64_t value;
        j.at("Value").get_to(value);
        pair = ParameterPair(parameterNames.back(), value);
    }
}

void to_json(json& j, const KernelConfiguration& configuration)
{
    j = json(configuration.GetPairs());
}

void from_json(const json& j, KernelConfiguration& configuration)
{
    auto pairs = j.get<std::vector<ParameterPair>>();
    configuration = KernelConfiguration(pairs);
}

void to_json(json& j, const KernelProfilingCounter& counter)
{
    j = json
    {
        {"Name", counter.GetName()},
        {"Type", counter.GetType()}
    };

    switch (counter.GetType())
    {
    case ProfilingCounterType::Int:
        j["Value"] = counter.GetValueInt();
        break;
    case ProfilingCounterType::UnsignedInt:
    case ProfilingCounterType::Throughput:
    case ProfilingCounterType::UtilizationLevel:
        j["Value"] = counter.GetValueUint();
        break;
    case ProfilingCounterType::Double:
    case ProfilingCounterType::Percent:
        j["Value"] = counter.GetValueDouble();
        break;
    default:
        KttError("Unhandled profiling counter type value");
    }
}

void from_json(const json& j, KernelProfilingCounter& counter)
{
    std::string name;
    j.at("Name").get_to(name);

    ProfilingCounterType type;
    j.at("Type").get_to(type);

    switch (type)
    {
    case ProfilingCounterType::Int:
        int64_t valueInt;
        j.at("Value").get_to(valueInt);
        counter = KernelProfilingCounter(name, type, valueInt);
        break;
    case ProfilingCounterType::UnsignedInt:
    case ProfilingCounterType::Throughput:
    case ProfilingCounterType::UtilizationLevel:
        uint64_t valueUint;
        j.at("Value").get_to(valueUint);
        counter = KernelProfilingCounter(name, type, valueUint);
        break;
    case ProfilingCounterType::Double:
    case ProfilingCounterType::Percent:
        double valueDouble;
        j.at("Value").get_to(valueDouble);
        counter = KernelProfilingCounter(name, type, valueDouble);
        break;
    default:
        KttError("Unhandled profiling counter type value");
    }
}

void to_json(json& j, const KernelCompilationData& data)
{
    j = json
    {
        {"MaxWorkGroupSize", data.m_MaxWorkGroupSize},
        {"LocalMemorySize", data.m_LocalMemorySize},
        {"PrivateMemorySize", data.m_PrivateMemorySize},
        {"ConstantMemorySize", data.m_ConstantMemorySize},
        {"RegistersCount", data.m_RegistersCount},
    };
}

void from_json(const json& j, KernelCompilationData& data)
{
    j.at("MaxWorkGroupSize").get_to(data.m_MaxWorkGroupSize);
    j.at("LocalMemorySize").get_to(data.m_LocalMemorySize);
    j.at("PrivateMemorySize").get_to(data.m_PrivateMemorySize);
    j.at("ConstantMemorySize").get_to(data.m_ConstantMemorySize);
    j.at("RegistersCount").get_to(data.m_RegistersCount);
}

void to_json(json& j, const KernelProfilingData& data)
{
    j = json
    {
        {"Counters", data.GetCounters()},
        {"RemainingProfilingRuns", data.GetRemainingProfilingRuns()}
    };
}

void from_json(const json& j, KernelProfilingData& data)
{
    uint64_t remainingRuns;
    j.at("RemainingProfilingRuns").get_to(remainingRuns);

    if (remainingRuns == 0)
    {
        data.SetCounters(j.at("Counters").get<std::vector<KernelProfilingCounter>>());
    }
    else
    {
        data = KernelProfilingData(remainingRuns);
    }
}

void to_json(json& j, const ComputationResult& result)
{
    const auto& time = TimeConfiguration::GetInstance();

    j = json
    {
        {"KernelFunction", result.GetKernelFunction()},
        {"Duration", time.ConvertFromNanosecondsDouble(result.GetDuration())},
        {"Overhead", time.ConvertFromNanosecondsDouble(result.GetOverhead())}
    };

    if (result.HasCompilationData())
    {
        j["CompilationData"] = result.GetCompilationData();
    }

    if (result.HasProfilingData())
    {
        j["ProfilingData"] = result.GetProfilingData();
    }
}

void from_json(const json& j, ComputationResult& result)
{
    std::string kernelFunction;
    j.at("KernelFunction").get_to(kernelFunction);
    result = ComputationResult(kernelFunction);

    const auto& time = TimeConfiguration::GetInstance();

    double duration;
    j.at("Duration").get_to(duration);
    const Nanoseconds durationNs = time.ConvertToNanosecondsDouble(duration);

    double overhead;
    j.at("Overhead").get_to(overhead);
    const Nanoseconds overheadNs = time.ConvertToNanosecondsDouble(overhead);

    result.SetDurationData(durationNs, overheadNs);

    if (j.contains("CompilationData"))
    {
        KernelCompilationData data;
        j.at("CompilationData").get_to(data);

        auto uniqueData = std::make_unique<KernelCompilationData>(data);
        result.SetCompilationData(std::move(uniqueData));
    }

    if (j.contains("ProfilingData"))
    {
        KernelProfilingData data;
        j.at("ProfilingData").get_to(data);

        auto uniqueData = std::make_unique<KernelProfilingData>(data);
        result.SetProfilingData(std::move(uniqueData));
    }
}

void to_json(json& j, const KernelResult& result)
{
    const auto& time = TimeConfiguration::GetInstance();

    j = json
    {
        {"KernelName", result.GetKernelName()},
        {"Status", result.GetStatus()},
        {"TotalDuration", time.ConvertFromNanosecondsDouble(result.GetTotalDuration())},
        {"TotalOverhead", time.ConvertFromNanosecondsDouble(result.GetTotalOverhead())},
        {"ExtraDuration", time.ConvertFromNanosecondsDouble(result.GetExtraDuration())},
        {"ExtraOverhead", time.ConvertFromNanosecondsDouble(result.GetExtraOverhead())},
        {"Configuration", result.GetConfiguration()},
        {"ComputationResults", result.GetResults()}
    };
}

void from_json(const json& j, KernelResult& result)
{
    std::string kernelName;
    j.at("KernelName").get_to(kernelName);

    KernelConfiguration configuration;
    j.at("Configuration").get_to(configuration);
    
    std::vector<ComputationResult> results;
    j.at("ComputationResults").get_to(results);

    result = KernelResult(kernelName, configuration, results);

    ResultStatus status;
    j.at("Status").get_to(status);
    result.SetStatus(status);

    const auto& time = TimeConfiguration::GetInstance();

    double extraDuration;
    j.at("ExtraDuration").get_to(extraDuration);
    const Nanoseconds extraDurationNs = time.ConvertToNanosecondsDouble(extraDuration);
    result.SetExtraDuration(extraDurationNs);

    double extraOverhead;
    j.at("ExtraOverhead").get_to(extraOverhead);
    const Nanoseconds extraOverheadNs = time.ConvertToNanosecondsDouble(extraOverhead);
    result.SetExtraOverhead(extraOverheadNs);
}

} // namespace ktt
