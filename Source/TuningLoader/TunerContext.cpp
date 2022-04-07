#include <TuningLoader/TunerContext.h>

namespace ktt
{

TunerContext::TunerContext() :
    m_Tuner(nullptr),
    m_DefinitionId(InvalidKernelDefinitionId),
    m_KernelId(InvalidKernelDefinitionId)
{}

void TunerContext::SetTuner(std::unique_ptr<Tuner> tuner)
{
    m_Tuner = std::move(tuner);
}

void TunerContext::SetKernelDefinitionId(const KernelDefinitionId id)
{
    m_DefinitionId = id;
}

void TunerContext::SetKernelId(const KernelId id)
{
    m_KernelId = id;
}

Tuner& TunerContext::GetTuner()
{
    return *m_Tuner;
}

KernelDefinitionId TunerContext::GetKernelDefinitionId() const
{
    return m_DefinitionId;
}

KernelId TunerContext::GetKernelId() const
{
    return m_KernelId;
}

std::unique_ptr<Tuner> TunerContext::RetrieveTuner()
{
    return std::move(m_Tuner);
}

} // namespace ktt
