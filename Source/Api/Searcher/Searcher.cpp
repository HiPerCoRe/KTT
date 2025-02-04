#include <Api/Searcher/Searcher.h>
#include <TuningRunner/ConfigurationData.h>

namespace ktt
{

void Searcher::OnInitialize()
{}

void Searcher::OnReset()
{}

Searcher::Searcher() :
    m_Data(nullptr)
{}

KernelConfiguration Searcher::GetConfiguration(const uint64_t index) const
{
    return m_Data->GetConfigurationForIndex(index);
}

uint64_t Searcher::GetIndex(const KernelConfiguration& configuration) const
{
    return m_Data->GetIndexForConfiguration(configuration);
}

KernelConfiguration Searcher::GetRandomConfiguration() const
{
    return m_Data->GetRandomConfiguration();
}

std::vector<KernelConfiguration> Searcher::GetNeighbourConfigurations(const KernelConfiguration& configuration,
    const uint64_t maxDifferences, const size_t maxNeighbours) const
{
    return m_Data->GetNeighbourConfigurations(configuration, maxDifferences, maxNeighbours);
}

uint64_t Searcher::GetConfigurationsCount() const
{
    return m_Data->GetTotalConfigurationsCount();
}

uint64_t Searcher::GetUnexploredConfigurationsCount() const
{
  return m_Data->GetTotalConfigurationsCount() - m_Data->GetExploredConfigurations().size();
}

const std::set<uint64_t>& Searcher::GetExploredIndices() const
{
    return m_Data->GetExploredConfigurations();
}

bool Searcher::IsInitialized() const
{
    return m_Data != nullptr;
}

void Searcher::Initialize(const ConfigurationData& data)
{
    m_Data = &data;
    OnInitialize();
}

void Searcher::Reset()
{
    OnReset();
    m_Data = nullptr;
}

} // namespace ktt
