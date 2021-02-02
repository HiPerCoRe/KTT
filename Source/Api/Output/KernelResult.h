#pragma once

#include <vector>

#include <Api/Output/ComputationResult.h>
#include <KttPlatform.h>
#include <KttTypes.h>

namespace ktt
{

class KTT_API KernelResult
{
public:
    KernelResult();
    explicit KernelResult(const KernelId id, const std::vector<ComputationResult>& results);

    void SetExtraDuration(const Nanoseconds duration);
    void SetExtraOverhead(const Nanoseconds overhead);

    KernelId GetId() const;
    const std::vector<ComputationResult>& GetResults() const;
    Nanoseconds GetTotalDuration() const;
    Nanoseconds GetTotalOverhead() const;
    bool IsValid() const;

private:
    std::vector<ComputationResult> m_Results;
    KernelId m_Id;
    Nanoseconds m_ExtraDuration;
    Nanoseconds m_ExtraOverhead;
};

} // namespace ktt
