#include <Api/Searcher/RandomSearcher.h>

namespace ktt
{

RandomSearcher::RandomSearcher() :
    Searcher(),
    m_CurrentIndex(std::numeric_limits<uint64_t>::max())
{}

void RandomSearcher::OnInitialize()
{
    m_CurrentIndex = GetRandomConfigurationIndex();
}

void RandomSearcher::OnReset()
{
    m_CurrentIndex = std::numeric_limits<uint64_t>::max();
    m_ExploredIndices.clear();
}

void RandomSearcher::CalculateNextConfiguration([[maybe_unused]] const KernelResult& previousResult)
{
    m_ExploredIndices.insert(m_CurrentIndex);
    m_CurrentIndex = GetRandomConfigurationIndex(m_ExploredIndices);
}

KernelConfiguration RandomSearcher::GetCurrentConfiguration() const
{
    return GetConfiguration(m_CurrentIndex);
}

} // namespace ktt
