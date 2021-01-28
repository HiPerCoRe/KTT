#include <Api/Output/KernelResult.h>

namespace ktt
{

KernelResult::KernelResult() :
    m_Id(InvalidKernelId)
{}

KernelResult::KernelResult(const KernelId id, const std::vector<ComputationResult>& results) :
    m_Id(id),
    m_Results(results)
{}

KernelId KernelResult::GetId() const
{
    return m_Id;
}

const std::vector<ComputationResult>& KernelResult::GetResults() const
{
    return m_Results;
}

bool KernelResult::IsValid() const
{
    return m_Id != InvalidKernelId;
}

} // namespace ktt
