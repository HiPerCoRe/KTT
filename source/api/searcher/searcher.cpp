#include <api/searcher/searcher.h>
#include <stdexcept>

namespace ktt
{

void Searcher::onInitialize()
{}

void Searcher::onReset()
{}

Searcher::Searcher() :
    configurations(nullptr)
{}

const std::vector<KernelConfiguration>& Searcher::getConfigurations() const
{
    return *configurations;
}

bool Searcher::isInitialized() const
{
    return configurations != nullptr;
}

void Searcher::initialize(const std::vector<KernelConfiguration>& configurations)
{
    if (configurations.empty())
    {
        throw std::runtime_error("No configurations provided for searcher");
    }

    this->configurations = &configurations;
    onInitialize();
}

void Searcher::reset()
{
    onReset();
    configurations = nullptr;
}

} // namespace ktt
