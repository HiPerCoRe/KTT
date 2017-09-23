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

TuningRunner::TuningRunner(ArgumentManager* argumentManager, KernelManager* kernelManager, Logger* logger, ComputeEngine* computeEngine) :
    argumentManager(argumentManager),
    kernelManager(kernelManager),
    logger(logger),
    computeEngine(computeEngine),
    resultValidator(argumentManager, kernelManager, logger, computeEngine),
    manipulatorInterfaceImplementation(std::make_unique<ManipulatorInterfaceImplementation>(computeEngine))
{}

std::vector<TuningResult> TuningRunner::tuneKernel(const size_t id)
{
    if (id >= kernelManager->getKernelCount())
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }

    std::vector<TuningResult> results;
    const Kernel* kernel = kernelManager->getKernel(id);
    resultValidator.computeReferenceResult(kernel);

    std::unique_ptr<Searcher> searcher = getSearcher(kernel->getSearchMethod(), kernel->getSearchArguments(),
        kernelManager->getKernelConfigurations(id, computeEngine->getCurrentDeviceInfo()), kernel->getParameters());
    size_t configurationsCount = searcher->getConfigurationsCount();

    for (size_t i = 0; i < configurationsCount; i++)
    {
        KernelConfiguration currentConfiguration = searcher->getNextConfiguration();
        TuningResult result(kernel->getName(), currentConfiguration);

        try
        {
            std::stringstream stream;
            stream << "Launching kernel <" << kernel->getName() << "> with configuration (" << i + 1 << " / " << configurationsCount << "): "
                << currentConfiguration;
            logger->log(stream.str());

            result = runKernel(kernel, currentConfiguration);
        }
        catch (const std::runtime_error& error)
        {
            logger->log(std::string("Kernel run failed, reason: ") + error.what() + "\n");
            results.emplace_back(kernel->getName(), currentConfiguration, std::string("Failed kernel run: ") + error.what());
        }

        searcher->calculateNextConfiguration(static_cast<double>(result.getTotalDuration()));
        if (validateResult(kernel, result))
        {
            results.push_back(result);
        }
        else
        {
            results.emplace_back(kernel->getName(), currentConfiguration, "Results differ");
        }

        computeEngine->clearBuffers(ArgumentAccessType::ReadWrite);
        computeEngine->clearBuffers(ArgumentAccessType::WriteOnly);

        auto manipulatorPointer = manipulatorMap.find(kernel->getId());
        if (manipulatorPointer != manipulatorMap.end())
        {
            computeEngine->clearBuffers(ArgumentAccessType::ReadOnly);
        }
    }

    computeEngine->clearBuffers();
    resultValidator.clearReferenceResults();
    return results;
}

void TuningRunner::runKernel(const size_t kernelId, const std::vector<ParameterValue>& kernelConfiguration)
{
    if (kernelId >= kernelManager->getKernelCount())
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(kernelId));
    }

    const Kernel* kernel = kernelManager->getKernel(kernelId);
    const KernelConfiguration launchConfiguration = kernelManager->getKernelConfiguration(kernelId, kernelConfiguration);

    try
    {
        runKernel(kernel, launchConfiguration);
    }
    catch (const std::runtime_error& error)
    {
        logger->log(std::string("Kernel run failed, reason: ") + error.what() + "\n");
    }

    computeEngine->clearBuffers();
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

TuningResult TuningRunner::runKernel(const Kernel* kernel, const KernelConfiguration& currentConfiguration)
{
    size_t kernelId = kernel->getId();
    std::string kernelName = kernel->getName();
    std::string source = kernelManager->getKernelSourceWithDefines(kernelId, currentConfiguration);

    auto manipulatorPointer = manipulatorMap.find(kernelId);
    if (manipulatorPointer != manipulatorMap.end())
    {
        auto kernelDataVector = getKernelDataVector(kernelId, KernelRuntimeData(kernelName, source, currentConfiguration.getGlobalSize(),
            currentConfiguration.getLocalSize(), kernel->getArgumentIndices()), manipulatorPointer->second->getUtilizedKernelIds(),
            currentConfiguration);
        logger->log("Launching tuning manipulator...");
        return runKernelWithManipulator(manipulatorPointer->second.get(), kernelDataVector, currentConfiguration);
    }

    KernelRunResult result = computeEngine->runKernel(source, kernel->getName(), convertDimensionVector(currentConfiguration.getGlobalSize()),
        convertDimensionVector(currentConfiguration.getLocalSize()), getKernelArgumentPointers(kernelId));
    return TuningResult(kernelName, currentConfiguration, result);
}

TuningResult TuningRunner::runKernelWithManipulator(TuningManipulator* manipulator,
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

    TuningResult tuningResult(kernelDataVector.at(0).second.getName(), currentConfiguration, result);
    tuningResult.setManipulatorDuration(manipulatorDuration);
    return tuningResult;
}

std::unique_ptr<Searcher> TuningRunner::getSearcher(const SearchMethod& searchMethod, const std::vector<double>& searchArguments,
    const std::vector<KernelConfiguration>& configurations, const std::vector<KernelParameter>& parameters) const
{
    std::unique_ptr<Searcher> searcher;

    switch (searchMethod)
    {
    case SearchMethod::FullSearch:
        searcher = std::make_unique<FullSearcher>(configurations);
        break;
    case SearchMethod::RandomSearch:
        searcher = std::make_unique<RandomSearcher>(configurations, searchArguments.at(0));
        break;
    case SearchMethod::PSO:
        searcher = std::make_unique<PSOSearcher>(configurations, parameters, searchArguments.at(0), static_cast<size_t>(searchArguments.at(1)),
            searchArguments.at(2), searchArguments.at(3), searchArguments.at(4));
        break;
    case SearchMethod::Annealing:
        searcher = std::make_unique<AnnealingSearcher>(configurations, searchArguments.at(0), searchArguments.at(1));
        break;
    default:
        throw std::runtime_error("Specified searcher is not supported");
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

bool TuningRunner::validateResult(const Kernel* kernel, const TuningResult& tuningResult)
{
    if (!tuningResult.isValid())
    {
        return false;
    }

    bool resultIsCorrect = resultValidator.validateArgumentsWithClass(kernel, tuningResult.getConfiguration());
    resultIsCorrect &= resultValidator.validateArgumentsWithKernel(kernel, tuningResult.getConfiguration());

    if (resultIsCorrect)
    {
        logger->log(std::string("Kernel run completed successfully in ") + std::to_string((tuningResult.getTotalDuration()) / 1'000'000)
            + "ms\n");
    }
    else
    {
        logger->log("Kernel run completed successfully, but results differ\n");
    }

    return resultIsCorrect;
}

} // namespace ktt
