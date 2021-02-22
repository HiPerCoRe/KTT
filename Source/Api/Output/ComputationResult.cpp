#include <limits>

#include <Api/Output/ComputationResult.h>
#include <Api/KttException.h>

namespace ktt
{

ComputationResult::ComputationResult() :
    m_Duration(InvalidDuration),
    m_Overhead(InvalidDuration)
{}

ComputationResult::ComputationResult(const std::string& kernelName, const std::string& configurationPrefix) :
    m_KernelName(kernelName),
    m_ConfigurationPrefix(configurationPrefix),
    m_Duration(InvalidDuration),
    m_Overhead(InvalidDuration)
{}

ComputationResult::ComputationResult(const ComputationResult& other) :
    m_KernelName(other.m_KernelName),
    m_ConfigurationPrefix(other.m_ConfigurationPrefix),
    m_Duration(other.m_Duration),
    m_Overhead(other.m_Overhead)
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

void ComputationResult::SetCompilationData(std::unique_ptr<KernelCompilationData> data)
{
    m_CompilationData = std::move(data);
}

void ComputationResult::SetProfilingData(std::unique_ptr<KernelProfilingData> data)
{
    m_ProfilingData = std::move(data);
}

const std::string& ComputationResult::GetKernelName() const
{
    return m_KernelName;
}

const std::string& ComputationResult::GetConfigurationPrefix() const
{
    return m_ConfigurationPrefix;
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

ComputationResult& ComputationResult::operator=(const ComputationResult& other)
{
    m_KernelName = other.m_KernelName;
    m_ConfigurationPrefix = other.m_ConfigurationPrefix;
    m_Duration = other.m_Duration;
    m_Overhead = other.m_Overhead;

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
