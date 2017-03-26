#include <string>

#include "tuning_runner.h"
#include "full_searcher.h"
#include "random_searcher.h"
#include "pso_searcher.h"

namespace ktt
{

TuningRunner::TuningRunner(ArgumentManager* argumentManager, KernelManager* kernelManager, OpenCLCore* openCLCore):
    argumentManager(argumentManager),
    kernelManager(kernelManager),
    openCLCore(openCLCore)
{}

std::vector<TuningResult> TuningRunner::tuneKernel(const size_t id)
{
    if (id >= kernelManager->getKernelCount())
    {
        throw std::runtime_error(std::string("Invalid kernel id: " + std::to_string(id)));
    }

    std::vector<TuningResult> results;

    if (kernelManager->getKernelConfigurations(id).size() == 1) // just single kernel run without any tuning
    {
        KernelRunResult result = openCLCore->runKernel(kernelManager->getKernel(id).getSource(), kernelManager->getKernel(id).getName(),
            convertDimensionVector(kernelManager->getKernel(id).getGlobalSize()), convertDimensionVector(kernelManager->getKernel(id).getLocalSize()),
            getKernelArguments(id));
        results.emplace_back(TuningResult(result.getDuration(), KernelConfiguration(kernelManager->getKernel(id).getGlobalSize(),
            kernelManager->getKernel(id).getLocalSize(), std::vector<ParameterValue>{})));
        return results;
    }

    std::unique_ptr<Searcher> searcher = getSearcher(kernelManager->getKernel(id).getSearchMethod(),
        kernelManager->getKernel(id).getSearchArguments(), kernelManager->getKernelConfigurations(id), kernelManager->getKernel(id).getParameters());

    size_t configurationsCount = searcher->getConfigurationsCount();
    for (size_t i = 0; i < configurationsCount; i++)
    {
        KernelConfiguration currentConfiguration = searcher->getNextConfiguration();
        std::string source = kernelManager->getKernelSourceWithDefines(id, currentConfiguration);

        KernelRunResult result = openCLCore->runKernel(source, kernelManager->getKernel(id).getName(),
            convertDimensionVector(currentConfiguration.getGlobalSize()), convertDimensionVector(currentConfiguration.getLocalSize()),
            getKernelArguments(id));
        
        searcher->calculateNextConfiguration(static_cast<double>(result.getDuration()));
        results.emplace_back(TuningResult(result.getDuration(), currentConfiguration));
    }

    return results;
}

std::unique_ptr<Searcher> TuningRunner::getSearcher(const SearchMethod& searchMethod, const std::vector<double>& searchArguments,
    const std::vector<KernelConfiguration>& configurations, const std::vector<KernelParameter>& parameters) const
{
    std::unique_ptr<Searcher> searcher;

    switch (searchMethod)
    {
    case SearchMethod::FullSearch:
        searcher = std::make_unique<FullSearcher>(configurations);
    case SearchMethod::RandomSearch:
        searcher = std::make_unique<RandomSearcher>(configurations, searchArguments.at(0));
    case SearchMethod::PSO:
        searcher = std::make_unique<PSOSearcher>(configurations, parameters, searchArguments.at(0), static_cast<size_t>(searchArguments.at(1)),
            searchArguments.at(2), searchArguments.at(3), searchArguments.at(4));
    default:
        throw std::runtime_error(std::string("Unsupported search method"));
    }

    return searcher;
}

std::vector<size_t> TuningRunner::convertDimensionVector(const DimensionVector& vector) const
{
    std::vector<size_t> result;

    result.push_back(std::get<0>(vector));
    result.push_back(std::get<1>(vector));
    result.push_back(std::get<2>(vector));

    return result;
}

std::vector<KernelArgument> TuningRunner::getKernelArguments(const size_t kernelId) const
{
    std::vector<KernelArgument> result;

    std::vector<size_t> argumentIndices = kernelManager->getKernel(kernelId).getArgumentIndices();
    
    for (const auto index : argumentIndices)
    {
        result.emplace_back(argumentManager->getArgument(index));
    }

    return result;
}

} // namespace ktt
