#include <Api/KttException.h>
#include <Api/Searcher/Searcher.h>
#include <TuningRunner/ConfigurationData.h>

namespace ktt
{

void Searcher::OnInitialize()
{}

void Searcher::OnReset()
{}

void Searcher::OnSeed([[maybe_unused]] const uint64_t seed)
{}

Searcher::Searcher() :
    m_Data(nullptr)
{}

Searcher::Searcher(const uint64_t seed) :
    m_Data(nullptr),
    m_Seed(seed)
{}

void Searcher::SetSeed(const uint64_t seed)
{
    m_Seed = seed;
}

bool Searcher::HasSeed() const
{
    return m_Seed.has_value();
}

uint64_t Searcher::GetSeed() const
{
    if (!HasSeed())
    {
        throw KttException("No seed was assigned to the searcher");
    }

    return m_Seed.value();
}

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

    if (HasSeed())
    {
        OnSeed(GetSeed());
    }

    OnInitialize();
}

void Searcher::Reset()
{
    OnReset();
    m_Data = nullptr;
}

} // namespace ktt
