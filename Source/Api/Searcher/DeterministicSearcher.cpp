#include <Api/Searcher/DeterministicSearcher.h>

namespace ktt
{

DeterministicSearcher::DeterministicSearcher() :
    Searcher(),
    m_Index(0)
{}

void DeterministicSearcher::OnReset()
{
    m_Index = 0;
}

void DeterministicSearcher::CalculateNextConfiguration([[maybe_unused]] const KernelResult& previousResult)
{
    ++m_Index;
}

KernelConfiguration DeterministicSearcher::GetCurrentConfiguration() const
{
    return GetConfiguration(m_Index);
}

} // namespace ktt
