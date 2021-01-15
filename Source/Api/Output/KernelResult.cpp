#include <limits>

#include <Api/Output/KernelResult.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/ErrorHandling/KttException.h>

namespace ktt
{

KernelResult::KernelResult() :
    m_Duration(InvalidDuration),
    m_Overhead(InvalidDuration),
    m_Status(KernelResultStatus::Failed)
{}

KernelResult::KernelResult(const std::string& kernelName, const std::string& configurationPrefix) :
    m_KernelName(kernelName),
    m_ConfigurationPrefix(configurationPrefix),
    m_Duration(InvalidDuration),
    m_Overhead(InvalidDuration),
    m_Status(KernelResultStatus::Failed)
{}

void KernelResult::SetStatus(const KernelResultStatus status)
{
    KttAssert(status != KernelResultStatus::Ok, "Status Ok should be set only by filling valid duration data");
    m_Status = status;
    m_Duration = InvalidDuration;
    m_Overhead = InvalidDuration;
}

void KernelResult::SetDurationData(const Nanoseconds duration, const Nanoseconds overhead)
{
    m_Duration = duration;
    m_Overhead = overhead;
    m_Status = KernelResultStatus::Ok;
}

void KernelResult::SetCompilationData(std::unique_ptr<KernelCompilationData> data)
{
    m_CompilationData = std::move(data);
}

void KernelResult::SetProfilingData(std::unique_ptr<KernelProfilingData> data)
{
    m_ProfilingData = std::move(data);
}

const std::string& KernelResult::GetKernelName() const
{
    return m_KernelName;
}

const std::string& KernelResult::GetConfigurationPrefix() const
{
    return m_ConfigurationPrefix;
}

Nanoseconds KernelResult::GetDuration() const
{
    return m_Duration;
}

Nanoseconds KernelResult::GetOverhead() const
{
    return m_Overhead;
}

KernelResultStatus KernelResult::GetStatus() const
{
    return m_Status;
}

bool KernelResult::IsValid() const
{
    return m_Status == KernelResultStatus::Ok;
}

bool KernelResult::HasCompilationData() const
{
    return m_CompilationData != nullptr;
}

const KernelCompilationData& KernelResult::GetCompilationData() const
{
    if (!HasCompilationData())
    {
        throw KttException("Kernel compilation data can only be retrieved after prior check that it exists");
    }

    return *m_CompilationData;
}

bool KernelResult::HasProfilingData() const
{
    return m_ProfilingData != nullptr;
}

const KernelProfilingData& KernelResult::GetProfilingData() const
{
    if (!HasProfilingData())
    {
        throw KttException("Kernel profiling data can only be retrieved after prior check that it exists");
    }

    return *m_ProfilingData;
}

} // namespace ktt
