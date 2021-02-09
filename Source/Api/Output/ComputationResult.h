#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <Api/Output/KernelCompilationData.h>
#include <Api/Output/KernelProfilingData.h>
#include <KttPlatform.h>
#include <KttTypes.h>

namespace ktt
{

class KTT_API ComputationResult
{
public:
    ComputationResult();
    explicit ComputationResult(const std::string& kernelName, const std::string& configurationPrefix);
    ComputationResult(const ComputationResult& other);

    void SetDurationData(const Nanoseconds duration, const Nanoseconds overhead);
    void SetCompilationData(std::unique_ptr<KernelCompilationData> data);
    void SetProfilingData(std::unique_ptr<KernelProfilingData> data);

    const std::string& GetKernelName() const;
    const std::string& GetConfigurationPrefix() const;
    Nanoseconds GetDuration() const;
    Nanoseconds GetOverhead() const;
    bool HasCompilationData() const;
    const KernelCompilationData& GetCompilationData() const;
    bool HasProfilingData() const;
    const KernelProfilingData& GetProfilingData() const;

    ComputationResult& operator=(const ComputationResult& other);

private:
    std::string m_KernelName;
    std::string m_ConfigurationPrefix;
    Nanoseconds m_Duration;
    Nanoseconds m_Overhead;
    std::unique_ptr<KernelCompilationData> m_CompilationData;
    std::unique_ptr<KernelProfilingData> m_ProfilingData;
};

} // namespace ktt
