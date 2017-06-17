#include <fstream>
#include <iterator>
#include <sstream>
#include <string>

#include "tuning_runner.h"
#include "searcher/annealing_searcher.h"
#include "searcher/full_searcher.h"
#include "searcher/pso_searcher.h"
#include "searcher/random_searcher.h"
#include "utility/ktt_utility.h"
#include "utility/timer.h"

namespace ktt
{

TuningRunner::TuningRunner(ArgumentManager* argumentManager, KernelManager* kernelManager, Logger* logger, ComputeApiDriver* computeApiDriver) :
    argumentManager(argumentManager),
    kernelManager(kernelManager),
    logger(logger),
    computeApiDriver(computeApiDriver),
    resultValidator(argumentManager, kernelManager, logger, computeApiDriver),
    manipulatorInterfaceImplementation(std::make_unique<ManipulatorInterfaceImplementation>(computeApiDriver))
{}

std::pair<std::vector<TuningResult>, std::vector<TuningResult>> TuningRunner::tuneKernel(const size_t id)
{
    if (id >= kernelManager->getKernelCount())
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }

    std::vector<TuningResult> results;
    std::vector<TuningResult> invalidResults;

    const Kernel* kernel = kernelManager->getKernel(id);
    resultValidator.computeReferenceResult(kernel);

    std::unique_ptr<Searcher> searcher = getSearcher(kernel->getSearchMethod(), kernel->getSearchArguments(),
        kernelManager->getKernelConfigurations(id, computeApiDriver->getCurrentDeviceInfo()), kernel->getParameters());
    size_t configurationsCount = searcher->getConfigurationsCount();

    for (size_t i = 0; i < configurationsCount; i++)
    {
        KernelConfiguration currentConfiguration = searcher->getNextConfiguration();
        KernelRunResult result;
        uint64_t manipulatorDuration = 0;

        try
        {
            auto resultPair = runKernel(kernel, currentConfiguration, i, configurationsCount);
            result = resultPair.first;
            manipulatorDuration = resultPair.second;
        }
        catch (const std::runtime_error& error)
        {
            logger->log(std::string("Kernel run failed, reason: ") + error.what() + "\n");
            invalidResults.push_back(TuningResult(kernel->getName(), currentConfiguration, std::string("Failed kernel run: ") + error.what()));
        }

        searcher->calculateNextConfiguration(static_cast<double>(result.getDuration() + manipulatorDuration));
        if (validateResult(kernel, result, manipulatorDuration, currentConfiguration))
        {
            results.emplace_back(TuningResult(kernel->getName(), result.getDuration(), manipulatorDuration, currentConfiguration));
        }
        else
        {
            invalidResults.push_back(TuningResult(kernel->getName(), currentConfiguration, "Results differ"));
        }

        computeApiDriver->clearBuffers(ArgumentMemoryType::ReadWrite);
        computeApiDriver->clearBuffers(ArgumentMemoryType::WriteOnly);

        auto manipulatorPointer = manipulatorMap.find(kernel->getId());
        if (manipulatorPointer != manipulatorMap.end())
        {
            computeApiDriver->clearBuffers(ArgumentMemoryType::ReadOnly);
        }
    }

    computeApiDriver->clearBuffers();
    resultValidator.clearReferenceResults();
    return std::make_pair(results, invalidResults);
}

void TuningRunner::setValidationMethod(const ValidationMethod& validationMethod, const double toleranceThreshold)
{
    resultValidator.setValidationMethod(validationMethod);
    resultValidator.setToleranceThreshold(toleranceThreshold);
}

void TuningRunner::setValidationRange(const size_t argumentId, const size_t validationRange)
{
    resultValidator.setValidationRange(argumentId, validationRange);
}

void TuningRunner::setReferenceKernel(const size_t kernelId, const size_t referenceKernelId,
    const std::vector<ParameterValue>& referenceKernelConfiguration, const std::vector<size_t>& resultArgumentIds)
{
    resultValidator.setReferenceKernel(kernelId, referenceKernelId, referenceKernelConfiguration, resultArgumentIds);
}

void TuningRunner::setReferenceClass(const size_t kernelId, std::unique_ptr<ReferenceClass> referenceClass,
    const std::vector<size_t>& resultArgumentIds)
{
    resultValidator.setReferenceClass(kernelId, std::move(referenceClass), resultArgumentIds);
}

void TuningRunner::setTuningManipulator(const size_t kernelId, std::unique_ptr<TuningManipulator> tuningManipulator)
{
    if (manipulatorMap.find(kernelId) != manipulatorMap.end())
    {
        manipulatorMap.erase(kernelId);
    }
    manipulatorMap.insert(std::make_pair(kernelId, std::move(tuningManipulator)));
}

void TuningRunner::enableArgumentPrinting(const size_t argumentId, const std::string& filePath, const ArgumentPrintCondition& argumentPrintCondition)
{
    resultValidator.enableArgumentPrinting(argumentId, filePath, argumentPrintCondition);
}

std::pair<KernelRunResult, uint64_t> TuningRunner::runKernel(const Kernel* kernel, const KernelConfiguration& currentConfiguration,
    const size_t currentConfigurationIndex, const size_t configurationsCount)
{
    KernelRunResult result;
    size_t kernelId = kernel->getId();
    std::string kernelName = kernel->getName();
    std::string source = kernelManager->getKernelSourceWithDefines(kernelId, currentConfiguration);
    std::stringstream stream;

    auto manipulatorPointer = manipulatorMap.find(kernelId);
    if (manipulatorPointer != manipulatorMap.end())
    {
        stream << "Launching kernel <" << kernelName << "> (manipulator detected) with configuration (" << currentConfigurationIndex + 1 << " / "
            << configurationsCount << "): " << currentConfiguration;
        logger->log(stream.str());
        auto kernelDataVector = getKernelDataVector(kernelId, KernelRuntimeData(kernelName, source, currentConfiguration.getGlobalSize(),
            currentConfiguration.getLocalSize(), kernel->getArgumentIndices()), manipulatorPointer->second->getUtilizedKernelIds(),
            currentConfiguration);
        return runKernelWithManipulator(manipulatorPointer->second.get(), kernelDataVector, currentConfiguration);
    }

    stream << "Launching kernel <" << kernelName << "> with configuration (" << currentConfigurationIndex + 1  << " / " << configurationsCount
        << "): " << currentConfiguration;
    logger->log(stream.str());
    result = computeApiDriver->runKernel(source, kernel->getName(), convertDimensionVector(currentConfiguration.getGlobalSize()),
        convertDimensionVector(currentConfiguration.getLocalSize()), getKernelArgumentPointers(kernelId));
    return std::make_pair(result, 0);
}

std::pair<KernelRunResult, uint64_t> TuningRunner::runKernelWithManipulator(TuningManipulator* manipulator,
    const std::vector<std::pair<size_t, KernelRuntimeData>>& kernelDataVector, const KernelConfiguration& currentConfiguration)
{
    manipulator->manipulatorInterface = manipulatorInterfaceImplementation.get();
    std::vector<const KernelArgument*> argumentPointers;
    for (const auto& kernelData : kernelDataVector)
    {
        manipulatorInterfaceImplementation->addKernel(kernelData.first, kernelData.second);
        auto currentKernelArguments = getKernelArgumentPointers(kernelData.first);
        for (const auto& argument : currentKernelArguments)
        {
            if (!elementExists(argument, argumentPointers))
            {
                argumentPointers.push_back(argument);
            }
        }
    }
    manipulatorInterfaceImplementation->setConfiguration(currentConfiguration);
    manipulatorInterfaceImplementation->setKernelArguments(argumentPointers);
    manipulatorInterfaceImplementation->uploadBuffers();

    Timer timer;
    try
    {
        timer.start();
        manipulator->launchComputation(kernelDataVector.at(0).first);
        timer.stop();
    }
    catch (const std::runtime_error&)
    {
        manipulatorInterfaceImplementation->clearData();
        manipulator->manipulatorInterface = nullptr;
        throw;
    }

    KernelRunResult result = manipulatorInterfaceImplementation->getCurrentResult();
    size_t manipulatorDuration = timer.getElapsedTime();
    manipulatorDuration -= result.getOverhead();

    manipulatorInterfaceImplementation->clearData();
    manipulator->manipulatorInterface = nullptr;
    return std::make_pair(result, manipulatorDuration);
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
        searcher.reset(new AnnealingSearcher(configurations, searchArguments.at(0), searchArguments.at(1)));
    }

    return searcher;
}

std::vector<KernelArgument> TuningRunner::getKernelArguments(const size_t kernelId) const
{
    std::vector<KernelArgument> result;

    std::vector<size_t> argumentIndices = kernelManager->getKernel(kernelId)->getArgumentIndices();
    
    for (const auto index : argumentIndices)
    {
        result.push_back(argumentManager->getArgument(index));
    }

    return result;
}

std::vector<const KernelArgument*> TuningRunner::getKernelArgumentPointers(const size_t kernelId) const
{
    std::vector<const KernelArgument*> result;

    std::vector<size_t> argumentIndices = kernelManager->getKernel(kernelId)->getArgumentIndices();
    
    for (const auto index : argumentIndices)
    {
        result.push_back(&argumentManager->getArgument(index));
    }

    return result;
}

std::vector<std::pair<size_t, KernelRuntimeData>> TuningRunner::getKernelDataVector(const size_t tunedKernelId,
    const KernelRuntimeData& tunedKernelData, const std::vector<std::pair<size_t, ThreadSizeUsage>>& additionalKernelData,
    const KernelConfiguration& currentConfiguration) const
{
    std::vector<std::pair<size_t, KernelRuntimeData>> result;
    result.push_back(std::make_pair(tunedKernelId, tunedKernelData));

    for (const auto& kernelDataPair : additionalKernelData)
    {
        if (kernelDataPair.first == tunedKernelId)
        {
            continue;
        }

        Kernel* kernel = kernelManager->getKernel(kernelDataPair.first);
        std::string source = kernelManager->getKernelSourceWithDefines(kernelDataPair.first, currentConfiguration);

        if (kernelDataPair.second == ThreadSizeUsage::Basic)
        {
            result.push_back(std::make_pair(kernelDataPair.first, KernelRuntimeData(kernel->getName(), source, kernel->getGlobalSize(),
                kernel->getLocalSize(), kernel->getArgumentIndices())));
        }
        else
        {
            KernelConfiguration configuration = kernelManager->getKernelConfiguration(kernelDataPair.first,
                currentConfiguration.getParameterValues());
            result.push_back(std::make_pair(kernelDataPair.first, KernelRuntimeData(kernel->getName(), source, configuration.getGlobalSize(),
                configuration.getLocalSize(), kernel->getArgumentIndices())));
        }
    }

    return result;
}

bool TuningRunner::validateResult(const Kernel* kernel, const KernelRunResult& result, const uint64_t manipulatorDuration,
    const KernelConfiguration& kernelConfiguration)
{
    if (!result.isValid())
    {
        return false;
    }

    bool resultIsCorrect = resultValidator.validateArgumentsWithClass(kernel, kernelConfiguration);
    resultIsCorrect &= resultValidator.validateArgumentsWithKernel(kernel, kernelConfiguration);

    if (resultIsCorrect)
    {
        logger->log(std::string("Kernel run completed successfully in ") + std::to_string((result.getDuration() + manipulatorDuration) / 1'000'000)
            + "ms\n");
    }
    else
    {
        logger->log("Kernel run completed successfully, but results differ\n");
    }

    return resultIsCorrect;
}

} // namespace ktt
