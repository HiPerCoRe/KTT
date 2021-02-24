#pragma once

#include <vector>

#include <Api/Configuration/KernelConfiguration.h>
#include <Api/Output/ComputationResult.h>
#include <Api/Output/ResultStatus.h>
#include <KttPlatform.h>
#include <KttTypes.h>

namespace ktt
{

class KTT_API KernelResult
{
public:
    KernelResult();
    explicit KernelResult(const std::string& kernelName, const KernelConfiguration& configuration);
    explicit KernelResult(const std::string& kernelName, const KernelConfiguration& configuration,
        const std::vector<ComputationResult>& results);

    void SetStatus(const ResultStatus status);
    void SetExtraDuration(const Nanoseconds duration);
    void SetExtraOverhead(const Nanoseconds overhead);

    const std::string& GetKernelName() const;
    const std::vector<ComputationResult>& GetResults() const;
    const KernelConfiguration& GetConfiguration() const;
    ResultStatus GetStatus() const;
    Nanoseconds GetKernelDuration() const;
    Nanoseconds GetKernelOverhead() const;
    Nanoseconds GetExtraDuration() const;
    Nanoseconds GetExtraOverhead() const;
    Nanoseconds GetTotalDuration() const;
    Nanoseconds GetTotalOverhead() const;
    bool IsValid() const;
    bool HasRemainingProfilingRuns() const;

private:
    KernelConfiguration m_Configuration;
    std::vector<ComputationResult> m_Results;
    std::string m_KernelName;
    Nanoseconds m_ExtraDuration;
    Nanoseconds m_ExtraOverhead;
    ResultStatus m_Status;
};

} // namespace ktt
