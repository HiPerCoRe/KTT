#include <Api/Searcher/Searcher.h>
#include <TuningRunner/ConfigurationTree.h>
#include <Utility/ErrorHandling/Assert.h>

namespace ktt
{

void Searcher::OnInitialize()
{}

void Searcher::OnReset()
{}

Searcher::Searcher() :
    m_Tree(nullptr)
{}

KernelConfiguration Searcher::GetConfiguration(const uint64_t index) const
{
    return m_Tree->GetConfiguration(index);
}

uint64_t Searcher::GetConfigurationsCount() const
{
    return m_Tree->GetConfigurationsCount();
}

bool Searcher::IsInitialized() const
{
    return m_Tree != nullptr;
}

void Searcher::Initialize(const ConfigurationTree& tree)
{
    KttAssert(tree.IsBuilt(), "Invalid configuration tree passed to searcher");

    m_Tree = &tree;
    OnInitialize();
}

void Searcher::Reset()
{
    OnReset();
    m_Tree = nullptr;
}

} // namespace ktt
