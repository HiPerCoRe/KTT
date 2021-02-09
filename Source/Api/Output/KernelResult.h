#pragma once

#include <vector>

#include <Api/Output/ComputationResult.h>
#include <Api/Output/ResultStatus.h>
#include <KttPlatform.h>
#include <KttTypes.h>

namespace ktt
{

class KTT_API KernelResult
{
public:
    KernelResult(const KernelId id);
    explicit KernelResult(const KernelId id, const std::vector<ComputationResult>& results);

    void SetStatus(const ResultStatus status);
    void SetExtraDuration(const Nanoseconds duration);
    void SetExtraOverhead(const Nanoseconds overhead);

    KernelId GetId() const;
    const std::vector<ComputationResult>& GetResults() const;
    ResultStatus GetStatus() const;
    Nanoseconds GetKernelDuration() const;
    Nanoseconds GetKernelOverhead() const;
    Nanoseconds GetTotalDuration() const;
    Nanoseconds GetTotalOverhead() const;
    bool IsValid() const;

private:
    std::vector<ComputationResult> m_Results;
    KernelId m_Id;
    Nanoseconds m_ExtraDuration;
    Nanoseconds m_ExtraOverhead;
    ResultStatus m_Status;
};

} // namespace ktt
