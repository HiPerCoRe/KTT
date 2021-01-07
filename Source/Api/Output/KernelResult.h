#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <Api/Output/KernelCompilationData.h>
#include <Api/Output/KernelProfilingData.h>
#include <Api/Output/KernelResultStatus.h>
#include <KttPlatform.h>

namespace ktt
{

class KTT_API KernelResult
{
public:
    KernelResult();
    explicit KernelResult(const std::string& kernelName, const std::string& configurationPrefix);

    void SetStatus(const KernelResultStatus status);
    void SetDurationData(const uint64_t duration, const uint64_t overhead);
    void SetCompilationData(std::unique_ptr<KernelCompilationData> data);
    void SetProfilingData(std::unique_ptr<KernelProfilingData> data);

    const std::string& GetKernelName() const;
    const std::string& GetConfigurationPrefix() const;
    uint64_t GetDuration() const;
    uint64_t GetOverhead() const;
    KernelResultStatus GetStatus() const;
    bool IsValid() const;
    bool HasCompilationData() const;
    const KernelCompilationData& GetCompilationData() const;
    bool HasProfilingData() const;
    const KernelProfilingData& GetProfilingData() const;

private:
    std::string m_KernelName;
    std::string m_ConfigurationPrefix;
    uint64_t m_Duration;
    uint64_t m_Overhead;
    std::unique_ptr<KernelCompilationData> m_CompilationData;
    std::unique_ptr<KernelProfilingData> m_ProfilingData;
    KernelResultStatus m_Status;
};

} // namespace ktt
