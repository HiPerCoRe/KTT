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

        switch (pair.GetValueType())
        {
            case ParameterValueType::Int:
                j[pair.GetName()] = std::get<int64_t>(pair.GetValue());
                break;
            case ParameterValueType::UnsignedInt:
                j[pair.GetName()] = pair.GetValueUint();
                break;
            case ParameterValueType::Double:
                j[pair.GetName()] = std::get<double>(pair.GetValue());
                break;
            case ParameterValueType::Bool:
                j[pair.GetName()] = std::get<bool>(pair.GetValue());
                break;
            case ParameterValueType::String:
                j[pair.GetName()] = pair.GetValueString();
                break;
            default:
                KttError("Unhandled parameter value type");
        }
    }
}

void from_json(const json& j, as_T4<KernelConfiguration>& configuration)
{
    std::vector<ParameterPair> pairs;
    for (auto it = j.begin(); it != j.end(); ++it) {
        std::string name = it.key();
        const auto &jsonValue = it.value();

        ParameterPair pair;

        if (jsonValue.is_boolean())
        {
            pair = ParameterPair(name, jsonValue.get<bool>());
        }
        else if (jsonValue.is_number_float())
        {
            pair = ParameterPair(name, jsonValue.get<double>());
        }
        else if (jsonValue.is_number_unsigned())
        {
            pair = ParameterPair(name, jsonValue.get<uint64_t>());
        }
        else if (jsonValue.is_number_integer())
        {
            pair = ParameterPair(name, jsonValue.get<int64_t>());
        }
        else if (jsonValue.is_string())
        {
            pair = ParameterPair(name, jsonValue.get<std::string>());
        }
        else
        {
            KttError("Unsupported parameter value type in configuration");
        }
        pairs.push_back(pair);
    }
    configuration.v = KernelConfiguration(pairs);
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
    j["times"]["compilation_time"] = time.ConvertFromNanosecondsDouble(result.v.GetCompilationOverhead());
    j["times"]["data"] = time.ConvertFromNanosecondsDouble(result.v.GetDataMovementOverhead());
    j["times"]["profiling_runs"] = time.ConvertFromNanosecondsDouble(result.v.GetProfilingRunsOverhead());
    j["times"]["profiling_overhead"] = time.ConvertFromNanosecondsDouble(result.v.GetProfilingOverhead());
    j["times"]["kernel_overhead"] = time.ConvertFromNanosecondsDouble(result.v.GetKernelOverhead());
    j["times"]["framework"] = time.ConvertFromNanosecondsDouble(result.v.GetTotalOverhead());
    j["times"]["search_algorithm"] = time.ConvertFromNanosecondsDouble(result.v.GetSearcherOverhead());
    j["times"]["validation"] = time.ConvertFromNanosecondsDouble(result.v.GetValidationOverhead());
    j["times"]["precise_measurement"] = time.ConvertFromNanosecondsDouble(result.v.GetPreciseMeasurementOverhead());
    j["times"]["runtimes"] = json::array({time.ConvertFromNanosecondsDouble(result.v.GetTotalDuration())});
    const ResultStatus& resultStatus = result.v.GetStatus();
    to_json(j["invalidity"], as_T4(resultStatus));
    j["correctness"] = correct;
    j["objectives"] = json::array({"time"});
    j["measurements"] = json::array();
    j["measurements"].push_back({{"name","time"}, {"value",time.ConvertFromNanosecondsDouble(result.v.GetTotalDuration())}, {"unit",""}});

    const std::vector<ComputationResult>& compResults = result.v.GetResults();
    if (!compResults.empty() && compResults[0].HasProfilingData()) {
        const std::vector<KernelProfilingCounter>& counters = compResults[0].GetProfilingData().GetCounters();
        for (const auto& counter : counters) {
            json j_counter = json::object();
            to_json(j_counter, as_T4(counter));
            j["measurements"].push_back(j_counter);
        }
    }

    // Add power measurements
    if (!compResults.empty()) {
        const auto& firstResult = compResults[0];
        
        if (firstResult.HasPowerData()) {
            j["measurements"].push_back({
                {"name", "power_usage"},
                {"value", firstResult.GetPowerUsage()},
                {"unit", "mW"}
            });
            
            j["measurements"].push_back({
                {"name", "energy_consumption"},
                {"value", firstResult.GetEnergyConsumption()},
                {"unit", "J"}
            });
        }
        
        if (firstResult.HasTemperatureData()) {
            j["measurements"].push_back({
                {"name", "temperature"},
                {"value", firstResult.GetTemperature()},
                {"unit", "°C"}
            });
        }
        
        if (firstResult.HasSMFrequencyData()) {
            j["measurements"].push_back({
                {"name", "sm_frequency"},
                {"value", firstResult.GetSMFrequency()},
                {"unit", "MHz"}
            });
        }
        
        if (firstResult.HasMemoryFrequencyData()) {
            j["measurements"].push_back({
                {"name", "memory_frequency"},
                {"value", firstResult.GetMemoryFrequency()},
                {"unit", "MHz"}
            });
        }
        
        if (firstResult.HasFanSpeedData()) {
            j["measurements"].push_back({
                {"name", "fan_speed"},
                {"value", firstResult.GetFanSpeed()},
                {"unit", "RPM"}
            });
        }
        
        if (firstResult.HasDurationStdevData()) {
            j["measurements"].push_back({
                {"name", "duration_stdev"},
                {"value", time.ConvertFromNanosecondsDouble(firstResult.GetDurationStdev())},
                {"unit", ""}
            });
        }
    }

    if (!compResults.empty() && compResults[0].HasCompilationData()) {
        const KernelCompilationData& compilationData =  compResults[0].GetCompilationData();
        const DimensionVector& globalSize = compResults[0].GetGlobalSize();
        const DimensionVector& localSize = compResults[0].GetLocalSize();
        json j_compilationData = json::object();
        j["compilation_data"] = {
            {"max_work_group_size", compilationData.m_MaxWorkGroupSize},
            {"local_memory_size", compilationData.m_LocalMemorySize},
            {"private_memory_size", compilationData.m_PrivateMemorySize},
            {"constant_memory_size", compilationData.m_ConstantMemorySize},
            {"registers", compilationData.m_RegistersCount},
            {"global_size", {
                {"x", globalSize.GetSizeX()},
                {"y", globalSize.GetSizeY()},
                {"z", globalSize.GetSizeZ()}
            }},
            {"local_size", {
                {"x", localSize.GetSizeX()},
                {"y", localSize.GetSizeY()},
                {"z", localSize.GetSizeZ()}
            }}
        };
    }
}

void from_json(const json& j, as_T4<KernelResult>& result)
{
    std::string kernelName = "";

    std::string timestamp;
    j.at("timestamp").get_to(timestamp);

    KernelConfiguration configuration;
    auto configurationWrapper = as_T4(configuration);
    from_json(j.at("configuration"), configurationWrapper);

    const auto& time = TimeConfiguration::GetInstance();

    std::vector<ComputationResult> results;
    ComputationResult computationResult;

    double duration;
    j.at("times").at("runtimes")[0].get_to(duration);
    const Nanoseconds durationNs = time.ConvertToNanosecondsDouble(duration);

    double compilationOverhead;
    j.at("times").at("compilation_time").get_to(compilationOverhead);
    const Nanoseconds compilationOverheadNs = time.ConvertToNanosecondsDouble(compilationOverhead);

    double dataMovementOverhead;
    if (j.at("times").contains("data"))
        j.at("times").at("data").get_to(dataMovementOverhead);
    else
        j.at("times").at("framework").get_to(dataMovementOverhead);
    const Nanoseconds dataMovementOverheadNs = time.ConvertToNanosecondsDouble(dataMovementOverhead);

    double kernelOverhead = 0.0;
    if (j.at("times").contains("kernel_overhead"))
        j.at("times").at("kernel_overhead").get_to(kernelOverhead);
    const Nanoseconds kernelOverheadNs = time.ConvertToNanosecondsDouble(kernelOverhead);

    double profilingRunsOverhead = 0.0;
    if (j.at("times").contains("profiling_runs"))
        j.at("times").at("profiling_runs").get_to(profilingRunsOverhead);
    const Nanoseconds profilingRunsOverheadNs = time.ConvertToNanosecondsDouble(profilingRunsOverhead);

    double profilingOverhead = 0.0;
    if (j.at("times").contains("profiling_overhead"))
        j.at("times").at("profiling_overhead").get_to(profilingOverhead);
    const Nanoseconds profilingOverheadNs = time.ConvertToNanosecondsDouble(profilingOverhead);

    double validationOverhead;
    j.at("times").at("validation").get_to(validationOverhead);
    const Nanoseconds validationOverheadNs = time.ConvertToNanosecondsDouble(validationOverhead);

    double preciseMeasurementOverhead = 0.0;
    if (j.at("times").contains("precise_measurement"))
        j.at("times").at("precise_measurement").get_to(preciseMeasurementOverhead);
    const Nanoseconds preciseMeasurementOverheadNs = time.ConvertToNanosecondsDouble(preciseMeasurementOverhead);

    // search_overhead is measured again in simulated tuning, so we are not deserializing it

    computationResult.SetDurationData(durationNs, kernelOverheadNs, compilationOverheadNs, profilingOverheadNs);
    if (j.at("measurements").size() > 1) {
        json j_measurements = j.at("measurements");
        //remove "time" measurement
        j_measurements.erase(j_measurements.begin());
        std::vector<KernelProfilingCounter> counters;

        for (const auto& j_measurement : j_measurements) {
            std::string name;
            j_measurement.at("name").get_to(name);
            
            // Check if this is a power measurement (not a profiling counter)
            if (name == "power_usage") {
                uint32_t value;
                j_measurement.at("value").get_to(value);
                computationResult.SetPowerUsage(value);
            } else if (name == "temperature") {
                double value;
                j_measurement.at("value").get_to(value);
                computationResult.SetTemperature(value);
            } else if (name == "sm_frequency") {
                uint32_t value;
                j_measurement.at("value").get_to(value);
                computationResult.SetSMFrequency(value);
            } else if (name == "memory_frequency") {
                uint32_t value;
                j_measurement.at("value").get_to(value);
                computationResult.SetMemoryFrequency(value);
            } else if (name == "fan_speed") {
                int32_t value;
                j_measurement.at("value").get_to(value);
                computationResult.SetFanSpeed(value);
            } else if (name == "duration_stdev") {
                double value;
                j_measurement.at("value").get_to(value);
                const Nanoseconds durationStdevNs = time.ConvertToNanosecondsDouble(value);
                computationResult.SetDurationStdev(durationStdevNs);
            } else if (name == "energy_consumption") {
                // Skip energy_consumption - it's computed, not stored
                continue;
            } else {
                // It's a profiling counter
                KernelProfilingCounter counter;
                auto counterWrapper = as_T4(counter);
                from_json(j_measurement, counterWrapper);
                counters.push_back(counter);
            }
        }
        
        if (!counters.empty()) {
            KernelProfilingData profilingData(counters);
            auto uniqueData = std::make_unique<KernelProfilingData>(profilingData);
            computationResult.SetProfilingData(std::move(uniqueData));
        }
    }

    if (j.contains("compilation_data"))
    {
        const auto& compilationDataJson = j["compilation_data"];

        if (!compilationDataJson.contains("max_work_group_size") ||
            !compilationDataJson.contains("local_memory_size") ||
            !compilationDataJson.contains("private_memory_size") ||
            !compilationDataJson.contains("constant_memory_size") ||
            !compilationDataJson.contains("registers") ||
            !compilationDataJson.contains("global_size") ||
            !compilationDataJson.contains("local_size"))
        {
            KttError(
                "Missing compilation data fields. Required fields: max_work_group_size, local_memory_size, private_memory_size, constant_memory_size, registers, global_size, local_size");
        }

        // Extract compilation data
        KernelCompilationData compData;
        compData.m_MaxWorkGroupSize = compilationDataJson["max_work_group_size"];
        compData.m_LocalMemorySize = compilationDataJson["local_memory_size"];
        compData.m_PrivateMemorySize = compilationDataJson["private_memory_size"];
        compData.m_ConstantMemorySize = compilationDataJson["constant_memory_size"];
        compData.m_RegistersCount = compilationDataJson["registers"];

        // Extract global size
        const auto& globalSizeJson = compilationDataJson["global_size"];
        if (!globalSizeJson.contains("x") || !globalSizeJson.contains("y") || !globalSizeJson.contains("z"))
        {
            KttError("Missing global_size dimensions");
        }
        DimensionVector globalSize(globalSizeJson["x"], globalSizeJson["y"], globalSizeJson["z"]);

        // Extract local size
        const auto& localSizeJson = compilationDataJson["local_size"];
        if (!localSizeJson.contains("x") || !localSizeJson.contains("y") || !localSizeJson.contains("z"))
        {
            KttError("Missing local_size dimensions");
        }
        DimensionVector localSize(localSizeJson["x"], localSizeJson["y"], localSizeJson["z"]);

        computationResult.SetCompilationData(std::make_unique<KernelCompilationData>(compData));
        computationResult.SetSizeData(globalSize, localSize);
    }

    results.push_back(computationResult);

    result.v = KernelResult(kernelName, configuration, results, timestamp);
    result.v.SetDataMovementOverhead(dataMovementOverheadNs);
    result.v.SetProfilingRunsOverhead(profilingRunsOverheadNs);
    result.v.SetValidationOverhead(validationOverheadNs);
    result.v.SetPreciseMeasurementOverhead(preciseMeasurementOverheadNs);

    ResultStatus status;
    auto statusWrapper = as_T4(status);
    from_json(j.at("invalidity"), statusWrapper);
    result.v.SetStatus(status);

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

void from_json(const json& j, as_T4<KernelProfilingCounter>& counter)
{
    std::string name;
    j.at("name").get_to(name);

    ProfilingCounterType type;
    j.at("type").get_to(type);

    switch (type)
    {
    case ProfilingCounterType::Int:
        int64_t valueInt;
        j.at("value").get_to(valueInt);
        counter.v = KernelProfilingCounter(name, type, valueInt);
        break;
    case ProfilingCounterType::UnsignedInt:
    case ProfilingCounterType::Throughput:
    case ProfilingCounterType::UtilizationLevel:
        uint64_t valueUint;
        j.at("value").get_to(valueUint);
        counter.v = KernelProfilingCounter(name, type, valueUint);
        break;
    case ProfilingCounterType::Double:
    case ProfilingCounterType::Percent:
        double valueDouble;
        j.at("value").get_to(valueDouble);
        counter.v = KernelProfilingCounter(name, type, valueDouble);
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

void from_json(const json& j, as_T4<TunerMetadata>& metadata)
{
    metadata.v.SetComputeApi(j.at("compute_api").get<ComputeApi>());
    metadata.v.SetPlatformName(j.at("platform").get<std::string>());
    metadata.v.SetDeviceName(j.at("device").get<std::string>());
    metadata.v.SetTimestamp(j.at("timestamp").get<std::string>());
    TimeUnit timeunit;
    auto wrapper = as_T4(timeunit);
    from_json(j.at("timeunit"), wrapper);
    metadata.v.SetTimeUnit(timeunit);
}

} // namespace ktt
