#include <limits>

#include <Api/Output/ComputationResult.h>
#include <Api/KttException.h>

namespace ktt
{

ComputationResult::ComputationResult() :
    m_Duration(InvalidDuration),
    m_Overhead(InvalidDuration)
{}

ComputationResult::ComputationResult(const std::string& kernelFunction) :
    m_KernelFunction(kernelFunction),
    m_Duration(InvalidDuration),
    m_Overhead(InvalidDuration)
{}

ComputationResult::ComputationResult(const ComputationResult& other) :
    m_KernelFunction(other.m_KernelFunction),
    m_GlobalSize(other.m_GlobalSize),
    m_LocalSize(other.m_LocalSize),
    m_Duration(other.m_Duration),
    m_Overhead(other.m_Overhead),
    m_PowerUsage(other.m_PowerUsage)
{
    if (other.HasCompilationData())
    {
        m_CompilationData = std::make_unique<KernelCompilationData>(*other.m_CompilationData);
    }

    if (other.HasProfilingData())
    {
        m_ProfilingData = std::make_unique<KernelProfilingData>(*other.m_ProfilingData);
    }
}

void ComputationResult::SetDurationData(const Nanoseconds duration, const Nanoseconds overhead)
{
    m_Duration = duration;
    m_Overhead = overhead;
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

uint32_t ComputationResult::GetPowerUsage() const
{
    if (!HasPowerData())
    {
        throw KttException("Power usage can only be retrieved after prior check that it exists");
    }

    return m_PowerUsage.value();
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
