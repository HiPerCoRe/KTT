#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include "tuning_runner.h"
#include "utility/ktt_utility.h"
#include "utility/timer.h"
#include "utility/result_loader.h"

namespace ktt
{

TuningRunner::TuningRunner(ArgumentManager* argumentManager, KernelManager* kernelManager, Logger* logger, ComputeEngine* computeEngine) :
    argumentManager(argumentManager),
    kernelManager(kernelManager),
    logger(logger),
    computeEngine(computeEngine),
    resultValidator(std::make_unique<ResultValidator>(argumentManager, kernelManager, logger, computeEngine)),
    manipulatorInterfaceImplementation(std::make_unique<ManipulatorInterfaceImplementation>(computeEngine))
{}

std::vector<TuningResult> TuningRunner::tuneKernel(const KernelId id)
{
    if (!kernelManager->isKernel(id))
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }

    const Kernel& kernel = kernelManager->getKernel(id);
    if (hasWritableZeroCopyArguments(kernel))
    {
        throw std::runtime_error("Kernel tuning cannot be performed with writable zero-copy arguments");
    }

    resultValidator->computeReferenceResult(kernel);
    configurationManager.clearData(id);
    configurationManager.setKernelConfigurations(id, kernelManager->getKernelConfigurations(id, computeEngine->getCurrentDeviceInfo()),
        kernel.getParameters());
    size_t configurationsCount = configurationManager.getConfigurationCount(id);
    std::vector<TuningResult> results;

    for (size_t i = 0; i < configurationsCount; i++)
    {
        KernelConfiguration currentConfiguration = configurationManager.getCurrentConfiguration(id);
        TuningResult result(kernel.getName(), currentConfiguration);

        try
        {
            std::stringstream stream;
            stream << "Launching kernel <" << kernel.getName() << "> with configuration (" << i + 1 << " / " << configurationsCount << "): "
                << currentConfiguration;
            logger->log(stream.str());

            if (kernel.hasTuningManipulator())
            {
                auto manipulatorPointer = tuningManipulators.find(id);
                result = runKernelWithManipulator(kernel, manipulatorPointer->second.get(), currentConfiguration,
                    std::vector<ArgumentOutputDescriptor>{});
            }
            else
            {
                result = runKernelSimple(kernel, currentConfiguration, std::vector<ArgumentOutputDescriptor>{});
            }
        }
        catch (const std::runtime_error& error)
        {
            logger->log(std::string("Kernel run failed, reason: ") + error.what() + "\n");
            results.emplace_back(kernel.getName(), currentConfiguration, std::string("Failed kernel run: ") + error.what());
        }

        configurationManager.calculateNextConfiguration(id, currentConfiguration, static_cast<double>(result.getTotalDuration()));
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
    configurationManager.clearSearcher(id);
    return results;
}

std::vector<TuningResult> TuningRunner::dryTuneKernel(const KernelId id, const std::string& filePath)
{
    if (!kernelManager->isKernel(id))
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }

    ResultLoader resultLoader;
    if (!resultLoader.loadResults(filePath))
    {
        throw std::runtime_error(std::string("Cannot read file: ") + filePath);
    }

    const Kernel& kernel = kernelManager->getKernel(id);

    configurationManager.clearData(id);
    configurationManager.setKernelConfigurations(id, kernelManager->getKernelConfigurations(id, computeEngine->getCurrentDeviceInfo()),
        kernel.getParameters());
    size_t configurationsCount = configurationManager.getConfigurationCount(id);
    std::vector<TuningResult> results;

    for (size_t i = 0; i < configurationsCount; i++)
    {
        KernelConfiguration currentConfiguration = configurationManager.getCurrentConfiguration(id);
        TuningResult result(kernel.getName(), currentConfiguration);

        try
        {
            std::stringstream stream;
            stream << "Launching kernel <" << kernel.getName() << "> with configuration (" << i + 1 << " / " << configurationsCount << "): "
                << currentConfiguration;
            logger->log(stream.str());

            //result = TuningResult(kernel.getName(), currentConfiguration, )
            result = resultLoader.readResult(currentConfiguration);
        }
        catch (const std::runtime_error& error)
        {
            logger->log(std::string("Kernel run failed, reason: ") + error.what() + "\n");
            results.emplace_back(kernel.getName(), currentConfiguration, std::string("Failed kernel run: ") + error.what());
        }

        configurationManager.calculateNextConfiguration(id, currentConfiguration, static_cast<double>(result.getTotalDuration()));
        results.push_back(result);
    }

    computeEngine->clearBuffers();
    resultValidator->clearReferenceResults();
    configurationManager.clearSearcher(id);
    return results;
}

std::vector<TuningResult> TuningRunner::tuneComposition(const KernelId id)
{
    if (!kernelManager->isComposition(id))
    {
        throw std::runtime_error(std::string("Invalid kernel composition id: ") + std::to_string(id));
    }

    const KernelComposition& composition = kernelManager->getKernelComposition(id);
    const Kernel compatibilityKernel = composition.transformToKernel();
    if (hasWritableZeroCopyArguments(compatibilityKernel))
    {
        throw std::runtime_error("Kernel composition tuning cannot be performed with writable zero-copy arguments");
    }

    resultValidator->computeReferenceResult(compatibilityKernel);
    configurationManager.clearData(id);
    configurationManager.setKernelConfigurations(id, kernelManager->getKernelCompositionConfigurations(id, computeEngine->getCurrentDeviceInfo()),
        composition.getParameters());
    size_t configurationsCount = configurationManager.getConfigurationCount(id);
    std::vector<TuningResult> results;

    for (size_t i = 0; i < configurationsCount; i++)
    {
        KernelConfiguration currentConfiguration = configurationManager.getCurrentConfiguration(id);
        TuningResult result(composition.getName(), currentConfiguration);

        try
        {
            std::stringstream stream;
            stream << "Launching kernel composition <" << composition.getName() << "> with configuration (" << i + 1 << " / " << configurationsCount
                << "): " << currentConfiguration;
            logger->log(stream.str());

            auto manipulatorPointer = tuningManipulators.find(id);
            result = runCompositionWithManipulator(composition, manipulatorPointer->second.get(), currentConfiguration,
                std::vector<ArgumentOutputDescriptor>{});
        }
        catch (const std::runtime_error& error)
        {
            logger->log(std::string("Kernel composition run failed, reason: ") + error.what() + "\n");
            results.emplace_back(composition.getName(), currentConfiguration, std::string("Failed kernel composition run: ") + error.what());
        }

        configurationManager.calculateNextConfiguration(id, currentConfiguration, static_cast<double>(result.getTotalDuration()));
        if (validateResult(compatibilityKernel, result))
        {
            results.push_back(result);
        }
        else
        {
            results.emplace_back(composition.getName(), currentConfiguration, "Results differ");
        }

        computeEngine->clearBuffers();
    }

    computeEngine->clearBuffers();
    resultValidator->clearReferenceResults();
    configurationManager.clearSearcher(id);
    return results;
}

TuningResult TuningRunner::tuneKernelByStep(const KernelId id, const std::vector<ArgumentOutputDescriptor>& output)
{
    if (!kernelManager->isKernel(id))
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }

    const Kernel& kernel = kernelManager->getKernel(id);
    resultValidator->computeReferenceResult(kernel);

    if (!configurationManager.hasKernelConfigurations(id))
    {
        configurationManager.setKernelConfigurations(id, kernelManager->getKernelConfigurations(id, computeEngine->getCurrentDeviceInfo()),
            kernel.getParameters());
    }

    KernelConfiguration currentConfiguration = configurationManager.getCurrentConfiguration(id);
    TuningResult result(kernel.getName(), currentConfiguration);

    try
    {
        std::stringstream stream;
        stream << "Launching kernel <" << kernel.getName() << "> with configuration: " << currentConfiguration;
        logger->log(stream.str());

        if (kernel.hasTuningManipulator())
        {
            auto manipulatorPointer = tuningManipulators.find(id);
            result = runKernelWithManipulator(kernel, manipulatorPointer->second.get(), currentConfiguration, output);
        }
        else
        {
            result = runKernelSimple(kernel, currentConfiguration, output);
        }
    }
    catch (const std::runtime_error& error)
    {
        logger->log(std::string("Kernel run failed, reason: ") + error.what() + "\n");
    }

    configurationManager.calculateNextConfiguration(id, currentConfiguration, static_cast<double>(result.getTotalDuration()));
    validateResult(kernel, result);

    computeEngine->clearBuffers();
    resultValidator->clearReferenceResults();
    return result;
}

TuningResult TuningRunner::tuneCompositionByStep(const KernelId id, const std::vector<ArgumentOutputDescriptor>& output)
{
    if (!kernelManager->isComposition(id))
    {
        throw std::runtime_error(std::string("Invalid kernel composition id: ") + std::to_string(id));
    }

    const KernelComposition& composition = kernelManager->getKernelComposition(id);
    const Kernel compatibilityKernel = composition.transformToKernel();
    resultValidator->computeReferenceResult(compatibilityKernel);

    if (!configurationManager.hasKernelConfigurations(id))
    {
        configurationManager.setKernelConfigurations(id, kernelManager->getKernelCompositionConfigurations(id,
            computeEngine->getCurrentDeviceInfo()), composition.getParameters());
    }

    KernelConfiguration currentConfiguration = configurationManager.getCurrentConfiguration(id);
    TuningResult result(composition.getName(), currentConfiguration);

    try
    {
        std::stringstream stream;
        stream << "Launching kernel composition <" << composition.getName() << "> with configuration: " << currentConfiguration;
        logger->log(stream.str());

        auto manipulatorPointer = tuningManipulators.find(id);
        result = runCompositionWithManipulator(composition, manipulatorPointer->second.get(), currentConfiguration, output);
    }
    catch (const std::runtime_error& error)
    {
        logger->log(std::string("Kernel composition run failed, reason: ") + error.what() + "\n");
    }

    configurationManager.calculateNextConfiguration(id, currentConfiguration, static_cast<double>(result.getTotalDuration()));
    validateResult(compatibilityKernel, result);

    computeEngine->clearBuffers();
    resultValidator->clearReferenceResults();
    return result;
}

void TuningRunner::runKernel(const KernelId id, const std::vector<ParameterPair>& configuration, const std::vector<ArgumentOutputDescriptor>& output)
{
    if (!kernelManager->isKernel(id))
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }

    const Kernel& kernel = kernelManager->getKernel(id);
    const KernelConfiguration launchConfiguration = kernelManager->getKernelConfiguration(id, configuration);

    std::stringstream stream;
    stream << "Running kernel <" << kernel.getName() << "> with configuration: " << launchConfiguration;
    logger->log(stream.str());

    try
    {
        if (kernel.hasTuningManipulator())
        {
            auto manipulatorPointer = tuningManipulators.find(id);
            runKernelWithManipulator(kernel, manipulatorPointer->second.get(), launchConfiguration, output);
        }
        else
        {
            runKernelSimple(kernel, launchConfiguration, output);
        }
    }
    catch (const std::runtime_error& error)
    {
        logger->log(std::string("Kernel run failed, reason: ") + error.what() + "\n");
    }

    computeEngine->clearBuffers();
}

void TuningRunner::runComposition(const KernelId id, const std::vector<ParameterPair>& configuration,
    const std::vector<ArgumentOutputDescriptor>& output)
{
    if (!kernelManager->isComposition(id))
    {
        throw std::runtime_error(std::string("Invalid kernel composition id: ") + std::to_string(id));
    }

    const KernelComposition& composition = kernelManager->getKernelComposition(id);
    const KernelConfiguration launchConfiguration = kernelManager->getKernelCompositionConfiguration(id, configuration);

    std::stringstream stream;
    stream << "Running kernel composition <" << composition.getName() << "> with configuration: " << launchConfiguration;
    logger->log(stream.str());

    try
    {
        auto manipulatorPointer = tuningManipulators.find(id);
        runCompositionWithManipulator(composition, manipulatorPointer->second.get(), launchConfiguration, output);
    }
    catch (const std::runtime_error& error)
    {
        logger->log(std::string("Kernel composition run failed, reason: ") + error.what() + "\n");
    }

    computeEngine->clearBuffers();
}

void TuningRunner::setSearchMethod(const SearchMethod& method, const std::vector<double>& arguments)
{
    configurationManager.setSearchMethod(method, arguments);
}

void TuningRunner::setValidationMethod(const ValidationMethod& method, const double toleranceThreshold)
{
    resultValidator->setValidationMethod(method);
    resultValidator->setToleranceThreshold(toleranceThreshold);
}

void TuningRunner::setValidationRange(const ArgumentId id, const size_t range)
{
    resultValidator->setValidationRange(id, range);
}

void TuningRunner::setArgumentComparator(const ArgumentId id, const std::function<bool(const void*, const void*)>& comparator)
{
    resultValidator->setArgumentComparator(id, comparator);
}

void TuningRunner::setReferenceKernel(const KernelId id, const KernelId referenceId, const std::vector<ParameterPair>& referenceConfiguration,
    const std::vector<ArgumentId>& validatedArgumentIds)
{
    resultValidator->setReferenceKernel(id, referenceId, referenceConfiguration, validatedArgumentIds);
}

void TuningRunner::setReferenceClass(const KernelId id, std::unique_ptr<ReferenceClass> referenceClass,
    const std::vector<ArgumentId>& validatedArgumentIds)
{
    resultValidator->setReferenceClass(id, std::move(referenceClass), validatedArgumentIds);
}

void TuningRunner::setTuningManipulator(const KernelId id, std::unique_ptr<TuningManipulator> manipulator)
{
    if (tuningManipulators.find(id) != tuningManipulators.end())
    {
        tuningManipulators.erase(id);
    }
    tuningManipulators.insert(std::make_pair(id, std::move(manipulator)));
}

std::vector<ParameterPair> TuningRunner::getBestConfiguration(const KernelId id) const
{
    return configurationManager.getBestConfiguration(id).getParameterPairs();
}

TuningResult TuningRunner::runKernelSimple(const Kernel& kernel, const KernelConfiguration& configuration,
    const std::vector<ArgumentOutputDescriptor>& output)
{
    KernelId kernelId = kernel.getId();
    std::string kernelName = kernel.getName();
    std::string source = kernelManager->getKernelSourceWithDefines(kernelId, configuration);

    KernelRuntimeData kernelData(kernelId, kernelName, source, configuration.getGlobalSize(), configuration.getLocalSize(), kernel.getArgumentIds());
    KernelRunResult result = computeEngine->runKernel(kernelData, argumentManager->getArguments(kernel.getArgumentIds()), output);
    return TuningResult(kernelName, configuration, result);
}

TuningResult TuningRunner::runKernelWithManipulator(const Kernel& kernel, TuningManipulator* manipulator, const KernelConfiguration& configuration,
    const std::vector<ArgumentOutputDescriptor>& output)
{
    KernelId kernelId = kernel.getId();
    std::string source = kernelManager->getKernelSourceWithDefines(kernelId, configuration);
    KernelRuntimeData kernelData(kernelId, kernel.getName(), source, configuration.getGlobalSize(), configuration.getLocalSize(),
        kernel.getArgumentIds());

    manipulator->manipulatorInterface = manipulatorInterfaceImplementation.get();
    manipulatorInterfaceImplementation->addKernel(kernelId, kernelData);
    manipulatorInterfaceImplementation->setConfiguration(configuration);
    manipulatorInterfaceImplementation->setKernelArguments(argumentManager->getArguments(kernel.getArgumentIds()));
    if (manipulator->enableArgumentPreload())
    {
        manipulatorInterfaceImplementation->uploadBuffers();
    }

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

    manipulatorInterfaceImplementation->downloadBuffers(output);
    KernelRunResult result = manipulatorInterfaceImplementation->getCurrentResult();
    size_t manipulatorDuration = timer.getElapsedTime();
    manipulatorDuration -= result.getOverhead();

    manipulatorInterfaceImplementation->clearData();
    manipulator->manipulatorInterface = nullptr;

    TuningResult tuningResult(kernel.getName(), configuration, result);
    tuningResult.setManipulatorDuration(manipulatorDuration);
    return tuningResult;
}

TuningResult TuningRunner::runCompositionWithManipulator(const KernelComposition& composition, TuningManipulator* manipulator,
    const KernelConfiguration& configuration, const std::vector<ArgumentOutputDescriptor>& output)
{
    manipulator->manipulatorInterface = manipulatorInterfaceImplementation.get();
    std::vector<KernelArgument*> allArguments = argumentManager->getArguments(composition.getSharedArgumentIds());

    for (const auto kernel : composition.getKernels())
    {
        KernelId kernelId = kernel->getId();
        std::vector<ArgumentId> argumentIds = composition.getKernelArgumentIds(kernelId);
        std::string source = kernelManager->getKernelSourceWithDefines(kernelId, configuration);

        KernelRuntimeData kernelData(kernelId, kernel->getName(), source, configuration.getCompositionKernelGlobalSize(kernelId),
            configuration.getCompositionKernelLocalSize(kernelId), argumentIds);
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

    manipulatorInterfaceImplementation->setConfiguration(configuration);
    manipulatorInterfaceImplementation->setKernelArguments(allArguments);
    if (manipulator->enableArgumentPreload())
    {
        manipulatorInterfaceImplementation->uploadBuffers();
    }

    Timer timer;
    try
    {
        timer.start();
        manipulator->launchComputation(composition.getId());
        timer.stop();
    }
    catch (const std::runtime_error&)
    {
        manipulatorInterfaceImplementation->clearData();
        manipulator->manipulatorInterface = nullptr;
        throw;
    }

    manipulatorInterfaceImplementation->downloadBuffers(output);
    KernelRunResult result = manipulatorInterfaceImplementation->getCurrentResult();
    size_t manipulatorDuration = timer.getElapsedTime();
    manipulatorDuration -= result.getOverhead();

    manipulatorInterfaceImplementation->clearData();
    manipulator->manipulatorInterface = nullptr;

    TuningResult tuningResult(composition.getName(), configuration, result);
    tuningResult.setManipulatorDuration(manipulatorDuration);
    return tuningResult;
}

bool TuningRunner::validateResult(const Kernel& kernel, const TuningResult& result)
{
    if (!result.isValid())
    {
        return false;
    }

    bool resultIsCorrect = resultValidator->validateArgumentsWithClass(kernel, result.getConfiguration());
    resultIsCorrect &= resultValidator->validateArgumentsWithKernel(kernel, result.getConfiguration());

    if (resultIsCorrect)
    {
        logger->log(std::string("Kernel run completed successfully in ") + std::to_string((result.getTotalDuration()) / 1'000'000)
            + "ms\n");
    }
    else
    {
        logger->log("Kernel run completed successfully, but results differ\n");
    }

    return resultIsCorrect;
}

bool TuningRunner::hasWritableZeroCopyArguments(const Kernel& kernel)
{
    std::vector<KernelArgument*> arguments = argumentManager->getArguments(kernel.getArgumentIds());

    for (const auto argument : arguments)
    {
        if (argument->getMemoryLocation() == ArgumentMemoryLocation::HostZeroCopy && argument->getAccessType() != ArgumentAccessType::ReadOnly)
        {
            return true;
        }
    }

    return false;
}

} // namespace ktt
