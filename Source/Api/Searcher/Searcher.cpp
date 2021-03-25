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

uint64_t Searcher::GetConfigurationsCount() const
{
    return m_Data->GetTotalConfigurationsCount();
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
