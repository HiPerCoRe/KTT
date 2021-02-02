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

const KernelConfiguration& DeterministicSearcher::GetCurrentConfiguration() const
{
    return GetConfigurations()[m_Index];
}

uint64_t DeterministicSearcher::GetUnexploredConfigurationCount() const
{
    if (m_Index >= GetConfigurations().size())
    {
        return 0;
    }

    return static_cast<uint64_t>(GetConfigurations().size() - m_Index);
}

} // namespace ktt
