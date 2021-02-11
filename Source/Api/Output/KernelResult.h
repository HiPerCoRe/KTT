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
    KernelResult(const KernelId id, const KernelConfiguration& configuration);
    explicit KernelResult(const KernelId id, const KernelConfiguration& configuration, const std::vector<ComputationResult>& results);

    void SetStatus(const ResultStatus status);
    void SetExtraDuration(const Nanoseconds duration);
    void SetExtraOverhead(const Nanoseconds overhead);

    KernelId GetId() const;
    const std::vector<ComputationResult>& GetResults() const;
    const KernelConfiguration& GetConfiguration() const;
    ResultStatus GetStatus() const;
    Nanoseconds GetKernelDuration() const;
    Nanoseconds GetKernelOverhead() const;
    Nanoseconds GetTotalDuration() const;
    Nanoseconds GetTotalOverhead() const;
    bool IsValid() const;
    bool HasRemainingProfilingRuns() const;

private:
    std::vector<ComputationResult> m_Results;
    KernelConfiguration m_Configuration;
    KernelId m_Id;
    Nanoseconds m_ExtraDuration;
    Nanoseconds m_ExtraOverhead;
    ResultStatus m_Status;
};

} // namespace ktt
