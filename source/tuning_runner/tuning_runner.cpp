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
    const Kernel& kernel = kernelManager->getKernel(id);
    std::unique_ptr<Searcher> searcher = getSearcher(kernel.getSearchMethod(), kernel.getSearchArguments(),
        kernelManager->getKernelConfigurations(id), kernel.getParameters());
    size_t configurationsCount = searcher->getConfigurationsCount();

    for (size_t i = 0; i < configurationsCount; i++)
    {
        KernelConfiguration currentConfiguration = searcher->getNextConfiguration();
        std::string source = kernelManager->getKernelSourceWithDefines(id, currentConfiguration);

        KernelRunResult result;
        try
        {
            result = openCLCore->runKernel(source, kernel.getName(), convertDimensionVector(currentConfiguration.getGlobalSize()),
                convertDimensionVector(currentConfiguration.getLocalSize()), getKernelArguments(id));
        }
        catch (const std::runtime_error& error)
        {
            std::cerr << "Kernel run execution failed for configuration: " << currentConfiguration << std::endl;
            std::cerr << error.what() << std::endl;
        }

        searcher->calculateNextConfiguration(static_cast<double>(result.getDuration()));
        if (result.getDuration() != 0)
        {
            results.emplace_back(TuningResult(kernel.getName(), result.getDuration(), currentConfiguration));
        }
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
        searcher.reset(new FullSearcher(configurations));
        break;
    case SearchMethod::RandomSearch:
        searcher.reset(new RandomSearcher(configurations, searchArguments.at(0)));
        break;
    case SearchMethod::PSO:
        searcher.reset(new PSOSearcher(configurations, parameters, searchArguments.at(0), static_cast<size_t>(searchArguments.at(1)),
            searchArguments.at(2), searchArguments.at(3), searchArguments.at(4)));
        break;
    default:
        throw std::runtime_error(std::string("Annealing searcher is currently not supported"));
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
