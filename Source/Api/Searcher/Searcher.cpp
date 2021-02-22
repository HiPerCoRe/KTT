#include <Api/Searcher/Searcher.h>
#include <Api/KttException.h>

namespace ktt
{

void Searcher::OnInitialize()
{}

void Searcher::OnReset()
{}

Searcher::Searcher() :
    m_Configurations(nullptr)
{}

const std::vector<KernelConfiguration>& Searcher::GetConfigurations() const
{
    return *m_Configurations;
}

bool Searcher::IsInitialized() const
{
    return m_Configurations != nullptr;
}

void Searcher::Initialize(const std::vector<KernelConfiguration>& configurations)
{
    if (configurations.empty())
    {
        throw KttException("No configurations provided for searcher");
    }

    m_Configurations = &configurations;
    OnInitialize();
}

void Searcher::Reset()
{
    OnReset();
    m_Configurations = nullptr;
}

} // namespace ktt
