#include <Output/TimeConfiguration/TimeConfiguration.h>
#include <Output/JsonConverters.h>
#include <Utility/ErrorHandling/Assert.h>

namespace ktt
{
    
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

void from_json(const json& /*j*/, ParameterPair& /*pair*/)
{
    // todo
}

void to_json(json& j, const KernelConfiguration& configuration)
{
    j = json(configuration.GetPairs());
}

void from_json(const json& /*j*/, KernelConfiguration& /*configuration*/)
{
    // todo
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

void from_json(const json& /*j*/, KernelProfilingCounter& /*counter*/)
{
    // todo
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

void from_json(const json& /*j*/, KernelProfilingData& /*data*/)
{
    // todo
}

void to_json(json& j, const ComputationResult& result)
{
    const auto& time = TimeConfiguration::GetInstance();

    j = json
    {
        {"KernelFunction", result.GetKernelFunction()},
        {"Duration", time.ConvertDuration(result.GetDuration())},
        {"Overhead", time.ConvertDuration(result.GetOverhead())}
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

void from_json(const json& /*j*/, ComputationResult& /*result*/)
{
    // todo
}

void to_json(json& j, const KernelResult& result)
{
    const auto& time = TimeConfiguration::GetInstance();

    j = json
    {
        {"KernelName", result.GetKernelName()},
        {"Status", result.GetStatus()},
        {"TotalDuration", time.ConvertDuration(result.GetTotalDuration())},
        {"TotalOverhead", time.ConvertDuration(result.GetTotalOverhead())},
        {"ExtraDuration", time.ConvertDuration(result.GetExtraDuration())},
        {"ExtraOverhead", time.ConvertDuration(result.GetExtraOverhead())},
        {"Configuration", result.GetConfiguration()},
        {"ComputationResults", result.GetResults()}
    };
}

void from_json(const json& /*j*/, KernelResult& /*result*/)
{
    // todo
}

} // namespace ktt
