#include<string>
#include <Output/TimeConfiguration/TimeConfiguration.h>
#include <Output/JsonT4Converters.h>
#include <Utility/ErrorHandling/Assert.h>

namespace ktt
{

void to_json(json& j, const as_T4<const KernelConfiguration>& configuration)
{
    j = json::object();
    const std::vector<ParameterPair>& pairs = configuration.v.GetPairs();
    for (const auto& pair : pairs) {
        std::string value;

        switch (pair.GetValueType())
        {
            case ParameterValueType::Int:
                value = std::to_string(std::get<int64_t>(pair.GetValue()));
                break;
            case ParameterValueType::UnsignedInt:
                value = std::to_string(pair.GetValueUint());
                break;
            case ParameterValueType::Double:
                value = std::to_string(std::get<double>(pair.GetValue()));
                break;
            case ParameterValueType::Bool:
                value = std::to_string(std::get<bool>(pair.GetValue()));
                break;
            case ParameterValueType::String:
                value = pair.GetValueString();
                break;
            default:
                KttError("Unhandled parameter value type");
        }

        j[pair.GetName()] = value;
    }
}
    
void to_json(json& j, const as_T4<const KernelResult>& result)
{
    const auto& time = TimeConfiguration::GetInstance();

    const auto& timestamp = result.v.GetTimestamp();

    const auto& configuration = result.v.GetConfiguration();

    uint correct = 0;
    if (result.v.GetStatus() == ResultStatus::ValidationFailed)
        correct = 0;
    else correct = 1;

    j = json::object();
    j["timestamp"] = timestamp;
    json j_configuration;
    to_json(j_configuration,as_T4(configuration));
    j["configuration"] = j_configuration;
    j["times"] = json::object();
    j["times"]["compilation"] = time.ConvertFromNanosecondsDouble(result.v.GetCompilationOverhead());
    j["times"]["data"] = time.ConvertFromNanosecondsDouble(result.v.GetDataMovementOverhead());
    j["times"]["profiling_runs"] = time.ConvertFromNanosecondsDouble(result.v.GetProfilingRunsOverhead());
    j["times"]["profiling_overhead"] = time.ConvertFromNanosecondsDouble(result.v.GetProfilingOverhead());
    j["times"]["kernel_overhead"] = time.ConvertFromNanosecondsDouble(result.v.GetKernelOverhead());
    j["times"]["framework"] = time.ConvertFromNanosecondsDouble(result.v.GetDataMovementOverhead()) + time.ConvertFromNanosecondsDouble(result.v.GetProfilingTotalOverhead()) + time.ConvertFromNanosecondsDouble(result.v.GetKernelOverhead());
    j["times"]["search_algorithm"] = time.ConvertFromNanosecondsDouble(result.v.GetSearcherOverhead());
    j["times"]["validation"] = time.ConvertFromNanosecondsDouble(result.v.GetValidationOverhead());
    j["times"]["runtimes"] = json::array({time.ConvertFromNanosecondsDouble(result.v.GetTotalDuration())});
    const ResultStatus& resultStatus = result.v.GetStatus();
    to_json(j["invalidity"], as_T4(resultStatus));
    j["correctness"] = correct;
    j["objectives"] = json::array({"time"});
    j["measurements"] = json::array();
    j["measurements"].push_back({{"name","time"}, {"value",time.ConvertFromNanosecondsDouble(result.v.GetTotalDuration())}, {"unit",""}});

    const std::vector<ComputationResult>& compResults = result.v.GetResults();
    if (compResults[0].HasProfilingData()) {
        const std::vector<KernelProfilingCounter>& counters = compResults[0].GetProfilingData().GetCounters();
        for (const auto& counter : counters) {
            json j_counter = json::object();
            to_json(j_counter, as_T4(counter));
            j["measurements"].push_back(j_counter);
        }
    }

}

void to_json(json& j, const as_T4<const KernelProfilingCounter>& counter)
{
    j = json
    {
        {"name", counter.v.GetName()},
        {"type", counter.v.GetTypeString()},
        {"unit", ""}
    };

    switch (counter.v.GetType())
    {
    case ProfilingCounterType::Int:
        j["value"] = counter.v.GetValueInt();
        break;
    case ProfilingCounterType::UnsignedInt:
    case ProfilingCounterType::Throughput:
    case ProfilingCounterType::UtilizationLevel:
        j["value"] = counter.v.GetValueUint();
        break;
    case ProfilingCounterType::Double:
    case ProfilingCounterType::Percent:
        j["value"] = counter.v.GetValueDouble();
        break;
    default:
        KttError("Unhandled profiling counter type value");
    }
}

void to_json(json& j, const as_T4<const TunerMetadata>& metadata)
{
    j = json
    {
        {"compute_api", metadata.v.GetComputeApi()},
        {"platform", metadata.v.GetPlatformName()},
        {"device", metadata.v.GetDeviceName()},
        {"autotuner", "KTT"},
        {"autotuner_version", metadata.v.GetKttVersion()},
        {"timestamp", metadata.v.GetTimestamp()}
    };
    json j_timeunit;
    const TimeUnit timeunit = metadata.v.GetTimeUnit();
    to_json(j_timeunit, as_T4(timeunit));
    j["timeunit"] = j_timeunit;
}
/*
void from_json(const json& j, TunerMetadata& metadata)
{
    metadata.SetComputeApi(j.at("ComputeApi").get<ComputeApi>());
    metadata.SetGlobalSizeType(j.at("GlobalSizeType").get<GlobalSizeType>());
    metadata.SetPlatformName(j.at("Platform").get<std::string>());
    metadata.SetDeviceName(j.at("Device").get<std::string>());
    metadata.SetKttVersion(j.at("KttVersion").get<std::string>());
    metadata.SetTimestamp(j.at("Timestamp").get<std::string>());
    metadata.SetTimeUnit(j.at("TimeUnit").get<TimeUnit>());
}

void to_json(json& j, const DimensionVector& vector)
{
    j = json
    {
        {"X", vector.GetSizeX()},
        {"Y", vector.GetSizeY()},
        {"Z", vector.GetSizeZ()}
    };
}

void from_json(const json& j, DimensionVector& vector)
{
    vector.SetSizeX(j.at("X").get<size_t>());
    vector.SetSizeY(j.at("Y").get<size_t>());
    vector.SetSizeZ(j.at("Z").get<size_t>());
}


void from_json(const json& j, ParameterPair& pair)
{
    std::string name;
    j.at("Name").get_to(name);

    ParameterValueType valueType;
    j.at("ValueType").get_to(valueType);

    switch (valueType)
    {
    case ParameterValueType::Int:
    {
        int64_t valueInt;
        j.at("Value").get_to(valueInt);
        pair = ParameterPair(name, valueInt);
        break;
    }
    case ParameterValueType::UnsignedInt:
    {
        uint64_t valueUint;
        j.at("Value").get_to(valueUint);
        pair = ParameterPair(name, valueUint);
        break;
    }
    case ParameterValueType::Double:
    {
        double valueDouble;
        j.at("Value").get_to(valueDouble);
        pair = ParameterPair(name, valueDouble);
        break;
    }
    case ParameterValueType::Bool:
    {
        bool valueBool;
        j.at("Value").get_to(valueBool);
        pair = ParameterPair(name, valueBool);
        break;
    }
    case ParameterValueType::String:
    {
        std::string valueString;
        j.at("Value").get_to(valueString);
        pair = ParameterPair(name, valueString);
        break;
    }
    default:
        KttError("Unhandled parameter value type");
    }
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
        {"Overhead", time.ConvertFromNanosecondsDouble(result.GetOverhead())},
        {"CompilationOverhead", time.ConvertFromNanosecondsDouble(result.GetCompilationOverhead())},
        {"GlobalSize", result.GetGlobalSize()},
        {"LocalSize", result.GetLocalSize()}
    };

    if (result.HasCompilationData())
    {
        j["CompilationData"] = result.GetCompilationData();
    }

    if (result.HasProfilingData())
    {
        j["ProfilingData"] = result.GetProfilingData();
    }

    if (result.HasPowerData())
    {
        j["PowerUsage"] = result.GetPowerUsage();
        j["EnergyConsumption"] = result.GetEnergyConsumption();
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

    double compilationOverhead;
    j.at("CompilationOverhead").get_to(compilationOverhead);
    const Nanoseconds overheadCompNs = time.ConvertToNanosecondsDouble(compilationOverhead);

    result.SetDurationData(durationNs, overheadNs, overheadCompNs);

    DimensionVector globalSize;
    j.at("GlobalSize").get_to(globalSize);

    DimensionVector localSize;
    j.at("LocalSize").get_to(localSize);

    result.SetSizeData(globalSize, localSize);

    if (j.contains("CompilationData"))
    {
        KernelCompilationData data;
        j.at("CompilationData").get_to(data);

        auto uniqueData = std::make_unique<KernelCompilationData>(data);
        result.SetCompilationData(std::move(uniqueData));
    };

    }

    if (j.contains("ProfilingData"))
    {
        KernelProfilingData data;
        j.at("ProfilingData").get_to(data);

        auto uniqueData = std::make_unique<KernelProfilingData>(data);
        result.SetProfilingData(std::move(uniqueData));
    }

    if (j.contains("PowerUsage"))
    {
        uint32_t powerUsage;
        j.at("PowerUsage").get_to(powerUsage);
        result.SetPowerUsage(powerUsage);
    }
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

    double dataMovementOverhead;
    j.at("DataMovementOverhead").get_to(dataMovementOverhead);
    const Nanoseconds dataMovementOverheadNs = time.ConvertToNanosecondsDouble(dataMovementOverhead);
    result.SetDataMovementOverhead(dataMovementOverheadNs);

    double validationOverhead;
    j.at("ValidationOverhead").get_to(validationOverhead);
    const Nanoseconds validationOverheadNs = time.ConvertToNanosecondsDouble(validationOverhead);
    result.SetValidationOverhead(validationOverheadNs);

    double searcherOverhead;
    j.at("SearcherOverhead").get_to(searcherOverhead);
    const Nanoseconds searcherOverheadNs = time.ConvertToNanosecondsDouble(searcherOverhead);
    result.SetSearcherOverhead(searcherOverheadNs);
}
*/
} // namespace ktt
