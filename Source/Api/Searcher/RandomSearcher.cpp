#include <Api/Searcher/RandomSearcher.h>

namespace ktt
{

RandomSearcher::RandomSearcher() :
    Searcher()
{}

void RandomSearcher::OnInitialize()
{
    m_CurrentConfiguration = GetRandomConfiguration();
}

bool RandomSearcher::CalculateNextConfiguration([[maybe_unused]] const KernelResult& previousResult)
{
    m_CurrentConfiguration = GetRandomConfiguration();
    return true;
}

KernelConfiguration RandomSearcher::GetCurrentConfiguration() const
{
    return m_CurrentConfiguration;
}

} // namespace ktt
