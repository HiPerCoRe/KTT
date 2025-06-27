#include <limits>

#include <Api/Output/ComputationResult.h>
#include <Api/KttException.h>

namespace ktt
{

ComputationResult::ComputationResult() :
    m_Duration(InvalidDuration),
    m_Overhead(InvalidDuration),
    m_CompilationOverhead(InvalidDuration)
{}

ComputationResult::ComputationResult(const std::string& kernelFunction) :
    m_KernelFunction(kernelFunction),
    m_Duration(InvalidDuration),
    m_Overhead(InvalidDuration),
    m_CompilationOverhead(InvalidDuration)
{}

ComputationResult::ComputationResult(const ComputationResult& other) :
    m_KernelFunction(other.m_KernelFunction),
    m_GlobalSize(other.m_GlobalSize),
    m_LocalSize(other.m_LocalSize),
    m_Duration(other.m_Duration),
    m_Overhead(other.m_Overhead),
    m_CompilationOverhead(other.m_CompilationOverhead),
    m_PowerUsage(other.m_PowerUsage),
    m_Temperature(other.m_Temperature),
    m_SMFrequency(other.m_SMFrequency),
    m_MemFrequency(other.m_MemFrequency)
{
    if (other.HasCompilationData())
    {
        m_CompilationData = std::make_unique<KernelCompilationData>(*other.m_CompilationData);
    }

    if (other.HasProfilingData())
    {
        m_ProfilingData = std::make_unique<KernelProfilingData>(*other.m_ProfilingData);
    }

    if (other.HasPowerData())
    {
        m_PowerUsage = other.GetPowerUsage();
    }

    if (other.HasTemperatureData())
    {
        m_Temperature = other.GetTemperature();
    }

    if (other.HasSMFrequencyData())
    {
        m_SMFrequency = other.GetSMFrequency();
    }

    if (other.HasMemoryFrequencyData())
    {
        m_MemFrequency = other.GetMemoryFrequency();
    }
}

void ComputationResult::SetDurationData(const Nanoseconds duration, const Nanoseconds overhead, const Nanoseconds compilationOverhead)
{
    m_Duration = duration;
    m_Overhead = overhead;
    m_CompilationOverhead = compilationOverhead;
}

void ComputationResult::SetSizeData(const DimensionVector& globalSize, const DimensionVector& localSize)
{
    m_GlobalSize = globalSize;
    m_LocalSize = localSize;
}

void ComputationResult::SetCompilationData(std::unique_ptr<KernelCompilationData> data)
{
    m_CompilationData = std::move(data);
}

void ComputationResult::SetProfilingData(std::unique_ptr<KernelProfilingData> data)
{
    m_ProfilingData = std::move(data);
}

void ComputationResult::SetPowerUsage(const uint32_t powerUsage)
{
    m_PowerUsage = powerUsage;
}

void ComputationResult::SetTemperature(const uint32_t temperature)
{
    m_Temperature = temperature;
}

void ComputationResult::SetSMFrequency(const uint32_t frequency)
{
    m_SMFrequency = frequency;
}

void ComputationResult::SetMemoryFrequency(const uint32_t frequency)
{
    m_MemFrequency = frequency;
}

const std::string& ComputationResult::GetKernelFunction() const
{
    return m_KernelFunction;
}

const DimensionVector& ComputationResult::GetGlobalSize() const
{
    return m_GlobalSize;
}

const DimensionVector& ComputationResult::GetLocalSize() const
{
    return m_LocalSize;
}

Nanoseconds ComputationResult::GetDuration() const
{
    return m_Duration;
}

Nanoseconds ComputationResult::GetOverhead() const
{
    return m_Overhead;
}

Nanoseconds ComputationResult::GetCompilationOverhead() const
{
    return m_CompilationOverhead;
}

bool ComputationResult::HasCompilationData() const
{
    return m_CompilationData != nullptr;
}

const KernelCompilationData& ComputationResult::GetCompilationData() const
{
    if (!HasCompilationData())
    {
        throw KttException("Kernel compilation data can only be retrieved after prior check that it exists");
    }

    return *m_CompilationData;
}

bool ComputationResult::HasProfilingData() const
{
    return m_ProfilingData != nullptr;
}

const KernelProfilingData& ComputationResult::GetProfilingData() const
{
    if (!HasProfilingData())
    {
        throw KttException("Kernel profiling data can only be retrieved after prior check that it exists");
    }

    return *m_ProfilingData;
}

bool ComputationResult::HasRemainingProfilingRuns() const
{
    if (!HasProfilingData())
    {
        return false;
    }

    return GetProfilingData().HasRemainingProfilingRuns();
}

bool ComputationResult::HasPowerData() const
{
    return m_PowerUsage.has_value();
}

bool ComputationResult::HasTemperatureData() const
{   
    return m_Temperature.has_value();
}

bool ComputationResult::HasSMFrequencyData() const
{
    return m_SMFrequency.has_value();
}

bool ComputationResult::HasMemoryFrequencyData() const
{
    return m_MemFrequency.has_value();
}

uint32_t ComputationResult::GetPowerUsage() const
{
    if (!HasPowerData())
    {
        throw KttException("Power usage can only be retrieved after prior check that it exists");
    }

    return m_PowerUsage.value();
}

uint32_t ComputationResult::GetTemperature() const
{   
    if (!HasTemperatureData())
    {
        throw KttException("Temperature can only be retrieved after prior check that it exists");
    }

    return m_Temperature.value();
}

uint32_t ComputationResult::GetSMFrequency() const
{
    if (!HasSMFrequencyData())
    {
        throw KttException("SM frequency can only be retrieved after prior check that it exists");
    }

    return m_SMFrequency.value();
}

uint32_t ComputationResult::GetMemoryFrequency() const
{
    if (!HasMemoryFrequencyData())
    {
        throw KttException("Memory frequency can only be retrieved after prior check that it exists");
    }

    return m_MemFrequency.value();
}

double ComputationResult::GetEnergyConsumption() const
{
    const double powerUsageWatts = static_cast<double>(GetPowerUsage()) / 1'000.0;
    const double durationSeconds = static_cast<double>(GetDuration()) / 1'000'000'000.0;
    return powerUsageWatts * durationSeconds;
}

ComputationResult& ComputationResult::operator=(const ComputationResult& other)
{
    m_KernelFunction = other.m_KernelFunction;
    m_Duration = other.m_Duration;
    m_Overhead = other.m_Overhead;
    m_GlobalSize = other.m_GlobalSize;
    m_LocalSize = other.m_LocalSize;
    m_PowerUsage = other.m_PowerUsage;
    m_Temperature = other.m_Temperature;
    m_SMFrequency = other.m_SMFrequency;
    m_MemFrequency = other.m_MemFrequency;

    if (other.HasCompilationData())
    {
        m_CompilationData = std::make_unique<KernelCompilationData>(*other.m_CompilationData);
    }

    if (other.HasProfilingData())
    {
        m_ProfilingData = std::make_unique<KernelProfilingData>(*other.m_ProfilingData);
    }

    return *this;
}

} // namespace ktt
