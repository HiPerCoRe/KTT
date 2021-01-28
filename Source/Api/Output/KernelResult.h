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

    KernelId GetId() const;
    const std::vector<ComputationResult>& GetResults() const;
    bool IsValid() const;

private:
    KernelId m_Id;
    std::vector<ComputationResult> m_Results;
};

} // namespace ktt
