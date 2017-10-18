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

TuningRunner::TuningRunner(ArgumentManager* argumentManager, KernelManager* kernelManager, Logger* logger, ComputeEngine* computeEngine,
    const RunMode& runMode) :
    argumentManager(argumentManager),
    kernelManager(kernelManager),
    logger(logger),
    computeEngine(computeEngine),
    resultValidator(nullptr),
    manipulatorInterfaceImplementation(std::make_unique<ManipulatorInterfaceImplementation>(computeEngine)),
    searchMethod(SearchMethod::FullSearch),
    runMode(runMode)
{
    if (runMode == RunMode::Tuning)
    {
        resultValidator = std::make_unique<ResultValidator>(argumentManager, kernelManager, logger, computeEngine);
    }
}

std::vector<TuningResult> TuningRunner::tuneKernel(const size_t id)
{
    if (runMode == RunMode::Computation)
    {
        throw std::runtime_error("Kernel tuning cannot be performed in computation mode");
    }

    if (!kernelManager->isKernel(id))
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }

    std::vector<TuningResult> results;
    const Kernel& kernel = kernelManager->getKernel(id);
    resultValidator->computeReferenceResult(kernel);

    std::unique_ptr<Searcher> searcher = getSearcher(searchMethod, searchArguments, kernelManager->getKernelConfigurations(id,
        computeEngine->getCurrentDeviceInfo()), kernel.getParameters());
    size_t configurationsCount = searcher->getConfigurationsCount();

    for (size_t i = 0; i < configurationsCount; i++)
    {
        KernelConfiguration currentConfiguration = searcher->getNextConfiguration();
        TuningResult result(kernel.getName(), currentConfiguration);

        try
        {
            std::stringstream stream;
            stream << "Launching kernel <" << kernel.getName() << "> with configuration (" << i + 1 << " / " << configurationsCount << "): "
                << currentConfiguration;
            logger->log(stream.str());

            if (kernel.hasTuningManipulator())
            {
                auto manipulatorPointer = manipulatorMap.find(id);
                result = runKernelWithManipulator(kernel, manipulatorPointer->second.get(), currentConfiguration, {});
            }
            else
            {
                result = runKernelSimple(kernel, currentConfiguration, {});
            }
        }
        catch (const std::runtime_error& error)
        {
            logger->log(std::string("Kernel run failed, reason: ") + error.what() + "\n");
            results.emplace_back(kernel.getName(), currentConfiguration, std::string("Failed kernel run: ") + error.what());
        }

        searcher->calculateNextConfiguration(static_cast<double>(result.getTotalDuration()));
        if (validateResult(kernel, result))
        {
            results.push_back(result);
        }
        else
        {
            results.emplace_back(kernel.getName(), currentConfiguration, "Results differ");
        }

        computeEngine->clearBuffers(ArgumentAccessType::ReadWrite);
        computeEngine->clearBuffers(ArgumentAccessType::WriteOnly);

        if (kernel.hasTuningManipulator())
        {
            computeEngine->clearBuffers(ArgumentAccessType::ReadOnly);
        }
    }

    computeEngine->clearBuffers();
    resultValidator->clearReferenceResults();
    return results;
}

std::vector<TuningResult> TuningRunner::tuneKernelComposition(const size_t id)
{
    if (runMode == RunMode::Computation)
    {
        throw std::runtime_error("Kernel tuning cannot be performed in computation mode");
    }

    if (!kernelManager->isKernelComposition(id))
    {
        throw std::runtime_error(std::string("Invalid kernel composition id: ") + std::to_string(id));
    }

    std::vector<TuningResult> results;
    const KernelComposition& composition = kernelManager->getKernelComposition(id);
    const Kernel& compatibilityKernel = compositionToKernel(composition);
    resultValidator->computeReferenceResult(compatibilityKernel);

    std::unique_ptr<Searcher> searcher = getSearcher(searchMethod, searchArguments, kernelManager->getKernelCompositionConfigurations(id,
        computeEngine->getCurrentDeviceInfo()), composition.getParameters());
    size_t configurationsCount = searcher->getConfigurationsCount();

    for (size_t i = 0; i < configurationsCount; i++)
    {
        KernelConfiguration currentConfiguration = searcher->getNextConfiguration();
        TuningResult result(composition.getName(), currentConfiguration);

        try
        {
            std::stringstream stream;
            stream << "Launching kernel composition <" << composition.getName() << "> with configuration (" << i + 1 << " / " << configurationsCount
                << "): " << currentConfiguration;
            logger->log(stream.str());

            auto manipulatorPointer = manipulatorMap.find(id);
            result = runKernelCompositionWithManipulator(composition, manipulatorPointer->second.get(), currentConfiguration, {});
        }
        catch (const std::runtime_error& error)
        {
            logger->log(std::string("Kernel composition run failed, reason: ") + error.what() + "\n");
            results.emplace_back(composition.getName(), currentConfiguration, std::string("Failed kernel composition run: ") + error.what());
        }

        searcher->calculateNextConfiguration(static_cast<double>(result.getTotalDuration()));
        if (validateResult(compatibilityKernel, result))
        {
            results.push_back(result);
        }
        else
        {
            results.emplace_back(composition.getName(), currentConfiguration, "Results differ");
        }

        computeEngine->clearBuffers(ArgumentAccessType::ReadWrite);
        computeEngine->clearBuffers(ArgumentAccessType::WriteOnly);
        computeEngine->clearBuffers(ArgumentAccessType::ReadOnly);
    }

    computeEngine->clearBuffers();
    resultValidator->clearReferenceResults();
    return results;
}

void TuningRunner::runKernel(const size_t kernelId, const std::vector<ParameterValue>& kernelConfiguration,
    const std::vector<ArgumentOutputDescriptor>& outputDescriptors)
{
    if (!kernelManager->isKernel(kernelId))
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(kernelId));
    }

    const Kernel& kernel = kernelManager->getKernel(kernelId);
    const KernelConfiguration launchConfiguration = kernelManager->getKernelConfiguration(kernelId, kernelConfiguration);

    std::stringstream stream;
    stream << "Running kernel <" << kernel.getName() << "> with configuration: " << launchConfiguration;
    logger->log(stream.str());

    try
    {
        if (kernel.hasTuningManipulator())
        {
            auto manipulatorPointer = manipulatorMap.find(kernelId);
            runKernelWithManipulator(kernel, manipulatorPointer->second.get(), launchConfiguration, outputDescriptors);
        }
        else
        {
            runKernelSimple(kernel, launchConfiguration, outputDescriptors);
        }
    }
    catch (const std::runtime_error& error)
    {
        logger->log(std::string("Kernel run failed, reason: ") + error.what() + "\n");
    }

    computeEngine->clearBuffers();
}

void TuningRunner::runKernelComposition(const size_t compositionId, const std::vector<ParameterValue>& compositionConfiguration,
    const std::vector<ArgumentOutputDescriptor>& outputDescriptors)
{
    if (!kernelManager->isKernelComposition(compositionId))
    {
        throw std::runtime_error(std::string("Invalid kernel composition id: ") + std::to_string(compositionId));
    }

    const KernelComposition& composition = kernelManager->getKernelComposition(compositionId);
    const KernelConfiguration launchConfiguration = kernelManager->getKernelCompositionConfiguration(compositionId, compositionConfiguration);

    std::stringstream stream;
    stream << "Running kernel composition <" << composition.getName() << "> with configuration: " << launchConfiguration;
    logger->log(stream.str());

    try
    {
        auto manipulatorPointer = manipulatorMap.find(compositionId);
        runKernelCompositionWithManipulator(composition, manipulatorPointer->second.get(), launchConfiguration, outputDescriptors);
    }
    catch (const std::runtime_error& error)
    {
        logger->log(std::string("Kernel composition run failed, reason: ") + error.what() + "\n");
    }

    computeEngine->clearBuffers();
}

void TuningRunner::setSearchMethod(const SearchMethod& searchMethod, const std::vector<double>& searchArguments)
{
    if (runMode == RunMode::Computation)
    {
        throw std::runtime_error("Kernel tuning cannot be performed in computation mode");
    }

    if (searchMethod == SearchMethod::RandomSearch && searchArguments.size() < 1
        || searchMethod == SearchMethod::Annealing && searchArguments.size() < 2
        || searchMethod == SearchMethod::PSO && searchArguments.size() < 5)
    {
        throw std::runtime_error(std::string("Insufficient number of arguments given for specified search method: ")
            + getSearchMethodName(searchMethod));
    }
    
    this->searchArguments = searchArguments;
    this->searchMethod = searchMethod;
}

void TuningRunner::setValidationMethod(const ValidationMethod& validationMethod, const double toleranceThreshold)
{
    if (runMode == RunMode::Computation)
    {
        throw std::runtime_error("Validation cannot be performed in computation mode");
    }
    resultValidator->setValidationMethod(validationMethod);
    resultValidator->setToleranceThreshold(toleranceThreshold);
}

void TuningRunner::setValidationRange(const size_t argumentId, const size_t validationRange)
{
    if (runMode == RunMode::Computation)
    {
        throw std::runtime_error("Validation cannot be performed in computation mode");
    }
    resultValidator->setValidationRange(argumentId, validationRange);
}

void TuningRunner::setReferenceKernel(const size_t kernelId, const size_t referenceKernelId,
    const std::vector<ParameterValue>& referenceKernelConfiguration, const std::vector<size_t>& resultArgumentIds)
{
    if (runMode == RunMode::Computation)
    {
        throw std::runtime_error("Validation cannot be performed in computation mode");
    }
    resultValidator->setReferenceKernel(kernelId, referenceKernelId, referenceKernelConfiguration, resultArgumentIds);
}

void TuningRunner::setReferenceClass(const size_t kernelId, std::unique_ptr<ReferenceClass> referenceClass,
    const std::vector<size_t>& resultArgumentIds)
{
    if (runMode == RunMode::Computation)
    {
        throw std::runtime_error("Validation cannot be performed in computation mode");
    }
    resultValidator->setReferenceClass(kernelId, std::move(referenceClass), resultArgumentIds);
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
    if (runMode == RunMode::Computation)
    {
        throw std::runtime_error("Argument printing cannot be performed in computation mode");
    }
    resultValidator->enableArgumentPrinting(argumentId, filePath, argumentPrintCondition);
}

TuningResult TuningRunner::runKernelSimple(const Kernel& kernel, const KernelConfiguration& currentConfiguration,
    const std::vector<ArgumentOutputDescriptor>& outputDescriptors)
{
    size_t kernelId = kernel.getId();
    std::string kernelName = kernel.getName();
    std::string source = kernelManager->getKernelSourceWithDefines(kernelId, currentConfiguration);

    KernelRuntimeData kernelData(kernelId, kernelName, source, currentConfiguration.getGlobalSize(), currentConfiguration.getLocalSize(),
        kernel.getArgumentIndices());
    KernelRunResult result = computeEngine->runKernel(kernelData, argumentManager->getArguments(kernel.getArgumentIndices()), outputDescriptors);
    return TuningResult(kernelName, currentConfiguration, result);
}

TuningResult TuningRunner::runKernelWithManipulator(const Kernel& kernel, TuningManipulator* manipulator,
    const KernelConfiguration& currentConfiguration, const std::vector<ArgumentOutputDescriptor>& outputDescriptors)
{
    size_t kernelId = kernel.getId();
    std::string source = kernelManager->getKernelSourceWithDefines(kernelId, currentConfiguration);
    KernelRuntimeData kernelData(kernelId, kernel.getName(), source, currentConfiguration.getGlobalSize(), currentConfiguration.getLocalSize(),
        kernel.getArgumentIndices());

    manipulator->manipulatorInterface = manipulatorInterfaceImplementation.get();
    manipulatorInterfaceImplementation->addKernel(kernelId, kernelData);
    manipulatorInterfaceImplementation->setConfiguration(currentConfiguration);
    manipulatorInterfaceImplementation->setKernelArguments(argumentManager->getArguments(kernel.getArgumentIndices()));
    manipulatorInterfaceImplementation->uploadBuffers();

    Timer timer;
    try
    {
        timer.start();
        manipulator->launchComputation(kernelId);
        timer.stop();
    }
    catch (const std::runtime_error&)
    {
        manipulatorInterfaceImplementation->clearData();
        manipulator->manipulatorInterface = nullptr;
        throw;
    }

    manipulatorInterfaceImplementation->downloadBuffers(outputDescriptors);
    KernelRunResult result = manipulatorInterfaceImplementation->getCurrentResult();
    size_t manipulatorDuration = timer.getElapsedTime();
    manipulatorDuration -= result.getOverhead();

    manipulatorInterfaceImplementation->clearData();
    manipulator->manipulatorInterface = nullptr;

    TuningResult tuningResult(kernel.getName(), currentConfiguration, result);
    tuningResult.setManipulatorDuration(manipulatorDuration);
    return tuningResult;
}

TuningResult TuningRunner::runKernelCompositionWithManipulator(const KernelComposition& kernelComposition, TuningManipulator* manipulator,
    const KernelConfiguration& currentConfiguration, const std::vector<ArgumentOutputDescriptor>& outputDescriptors)
{
    manipulator->manipulatorInterface = manipulatorInterfaceImplementation.get();
    std::vector<KernelArgument*> allArguments = argumentManager->getArguments(kernelComposition.getSharedArgumentIds());

    for (const auto kernel : kernelComposition.getKernels())
    {
        size_t kernelId = kernel->getId();
        std::vector<size_t> argumentIds = kernelComposition.getKernelArgumentIds(kernelId);
        std::string source = kernelManager->getKernelSourceWithDefines(kernelId, currentConfiguration);

        KernelRuntimeData kernelData(kernelId, kernel->getName(), source, currentConfiguration.getGlobalSize(kernelId),
            currentConfiguration.getLocalSize(kernelId), argumentIds);
        manipulatorInterfaceImplementation->addKernel(kernelId, kernelData);

        std::vector<KernelArgument*> newArguments = argumentManager->getArguments(argumentIds);
        for (const auto newArgument : newArguments)
        {
            if (!elementExists(newArgument, allArguments))
            {
                allArguments.push_back(newArgument);
            }
        }
    }

    manipulatorInterfaceImplementation->setConfiguration(currentConfiguration);
    manipulatorInterfaceImplementation->setKernelArguments(allArguments);
    manipulatorInterfaceImplementation->uploadBuffers();

    Timer timer;
    try
    {
        timer.start();
        manipulator->launchComputation(kernelComposition.getId());
        timer.stop();
    }
    catch (const std::runtime_error&)
    {
        manipulatorInterfaceImplementation->clearData();
        manipulator->manipulatorInterface = nullptr;
        throw;
    }

    manipulatorInterfaceImplementation->downloadBuffers(outputDescriptors);
    KernelRunResult result = manipulatorInterfaceImplementation->getCurrentResult();
    size_t manipulatorDuration = timer.getElapsedTime();
    manipulatorDuration -= result.getOverhead();

    manipulatorInterfaceImplementation->clearData();
    manipulator->manipulatorInterface = nullptr;

    TuningResult tuningResult(kernelComposition.getName(), currentConfiguration, result);
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

bool TuningRunner::validateResult(const Kernel& kernel, const TuningResult& tuningResult)
{
    if (runMode == RunMode::Computation)
    {
        throw std::runtime_error("Validation cannot be performed in computation mode");
    }

    if (!tuningResult.isValid())
    {
        return false;
    }

    bool resultIsCorrect = resultValidator->validateArgumentsWithClass(kernel, tuningResult.getConfiguration());
    resultIsCorrect &= resultValidator->validateArgumentsWithKernel(kernel, tuningResult.getConfiguration());

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

std::string TuningRunner::getSearchMethodName(const SearchMethod& searchMethod) const
{
    switch (searchMethod)
    {
    case SearchMethod::FullSearch:
        return std::string("FullSearch");
    case SearchMethod::RandomSearch:
        return std::string("RandomSearch");
    case SearchMethod::PSO:
        return std::string("PSO");
    case SearchMethod::Annealing:
        return std::string("Annealing");
    default:
        return std::string("Unknown search method");
    }
}

Kernel TuningRunner::compositionToKernel(const KernelComposition& composition) const
{
    Kernel kernel(composition.getId(), "", composition.getName(), DimensionVector(0, 0, 0), DimensionVector(0, 0, 0));
    kernel.setTuningManipulatorFlag(true);

    for (const auto& constraint : composition.getConstraints())
    {
        kernel.addConstraint(constraint);
    }

    for (const auto& parameter : composition.getParameters())
    {
        kernel.addParameter(parameter);
    }

    std::vector<size_t> argumentIds;
    for (const auto id : composition.getSharedArgumentIds())
    {
        argumentIds.push_back(id);
    }

    for (const auto& kernel : composition.getKernels())
    {
        for (const auto id : composition.getKernelArgumentIds(kernel->getId()))
        {
            argumentIds.push_back(id);
        }
    }
    kernel.setArguments(argumentIds);

    return kernel;
}

} // namespace ktt
