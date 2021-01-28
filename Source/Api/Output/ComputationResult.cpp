#include <limits>

#include <Api/Output/ComputationResult.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/ErrorHandling/KttException.h>

namespace ktt
{

ComputationResult::ComputationResult() :
    m_Duration(InvalidDuration),
    m_Overhead(InvalidDuration),
    m_Status(ComputationStatus::Failed)
{}

ComputationResult::ComputationResult(const std::string& kernelName, const std::string& configurationPrefix) :
    m_KernelName(kernelName),
    m_ConfigurationPrefix(configurationPrefix),
    m_Duration(InvalidDuration),
    m_Overhead(InvalidDuration),
    m_Status(ComputationStatus::Failed)
{}

ComputationResult::ComputationResult(const ComputationResult& other) :
    m_KernelName(other.m_KernelName),
    m_ConfigurationPrefix(other.m_ConfigurationPrefix),
    m_Duration(other.m_Duration),
    m_Overhead(other.m_Overhead),
    m_Status(other.m_Status)
{
    m_CompilationData = std::make_unique<KernelCompilationData>(*other.m_CompilationData);
    m_ProfilingData = std::make_unique<KernelProfilingData>(*other.m_ProfilingData);
}

void ComputationResult::SetStatus(const ComputationStatus status)
{
    KttAssert(status != ComputationStatus::Ok, "Status Ok should be set only by filling valid duration data");
    m_Status = status;
    m_Duration = InvalidDuration;
    m_Overhead = InvalidDuration;
}

void ComputationResult::SetDurationData(const Nanoseconds duration, const Nanoseconds overhead)
{
    m_Duration = duration;
    m_Overhead = overhead;
    m_Status = ComputationStatus::Ok;
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

ComputationStatus ComputationResult::GetStatus() const
{
    return m_Status;
}

bool ComputationResult::IsValid() const
{
    return m_Status == ComputationStatus::Ok;
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

ComputationResult& ComputationResult::operator=(const ComputationResult& other)
{
    m_KernelName = other.m_KernelName;
    m_ConfigurationPrefix = other.m_ConfigurationPrefix;
    m_Duration = other.m_Duration;
    m_Overhead = other.m_Overhead;
    m_Status = other.m_Status;
    m_CompilationData = std::make_unique<KernelCompilationData>(*other.m_CompilationData);
    m_ProfilingData = std::make_unique<KernelProfilingData>(*other.m_ProfilingData);
    return *this;
}

} // namespace ktt
