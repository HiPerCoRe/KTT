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

TuningRunner::TuningRunner(ArgumentManager* argumentManager, KernelManager* kernelManager, KernelRunner* kernelRunner, Logger* logger) :
    argumentManager(argumentManager),
    kernelManager(kernelManager),
    kernelRunner(kernelRunner),
    logger(logger),
    resultValidator(std::make_unique<ResultValidator>(argumentManager, kernelRunner, logger))
{}

std::vector<KernelResult> TuningRunner::tuneKernel(const KernelId id)
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
    configurationManager.setKernelConfigurations(id, kernelManager->getKernelConfigurations(id), kernel.getParameters());
    size_t configurationsCount = configurationManager.getConfigurationCount(id);
    std::vector<KernelResult> results;

    for (size_t i = 0; i < configurationsCount; i++)
    {
        std::stringstream stream;
        stream << "Launching configuration: " << i + 1 << " / " << configurationsCount;
        logger->log(stream.str());

        KernelConfiguration currentConfiguration = configurationManager.getCurrentConfiguration(id);
        KernelResult result = kernelRunner->runKernel(id, currentConfiguration.getParameterPairs(), std::vector<ArgumentOutputDescriptor>{});

        configurationManager.calculateNextConfiguration(id, currentConfiguration, static_cast<double>(result.getComputationDuration()));
        if (validateResult(kernel, result))
        {
            results.push_back(result);
        }
        else
        {
            results.emplace_back(kernel.getName(), currentConfiguration, "Results differ");
        }

        kernelRunner->clearBuffers(ArgumentAccessType::ReadWrite);
        kernelRunner->clearBuffers(ArgumentAccessType::WriteOnly);
        if (kernel.hasTuningManipulator())
        {
            kernelRunner->clearBuffers(ArgumentAccessType::ReadOnly);
        }
    }

    kernelRunner->clearBuffers();
    resultValidator->clearReferenceResults();
    configurationManager.clearSearcher(id);
    return results;
}

std::vector<KernelResult> TuningRunner::dryTuneKernel(const KernelId id, const std::string& filePath)
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
    configurationManager.setKernelConfigurations(id, kernelManager->getKernelConfigurations(id), kernel.getParameters());
    size_t configurationsCount = configurationManager.getConfigurationCount(id);
    std::vector<KernelResult> results;

    for (size_t i = 0; i < configurationsCount; i++)
    {
        KernelConfiguration currentConfiguration = configurationManager.getCurrentConfiguration(id);
        KernelResult result(kernel.getName(), currentConfiguration);

        try
        {
            std::stringstream stream;
            stream << "Launching kernel <" << kernel.getName() << "> with configuration (" << i + 1 << " / " << configurationsCount << "): "
                << currentConfiguration;
            logger->log(stream.str());

            result = resultLoader.readResult(currentConfiguration);
            result.setConfiguration(currentConfiguration);
        }
        catch (const std::runtime_error& error)
        {
            logger->log(std::string("Kernel run failed, reason: ") + error.what() + "\n");
            results.emplace_back(kernel.getName(), currentConfiguration, std::string("Failed kernel run: ") + error.what());
        }

        configurationManager.calculateNextConfiguration(id, currentConfiguration, static_cast<double>(result.getComputationDuration()));
        results.push_back(result);
    }

    kernelRunner->clearBuffers();
    resultValidator->clearReferenceResults();
    configurationManager.clearSearcher(id);
    return results;
}

std::vector<KernelResult> TuningRunner::tuneComposition(const KernelId id)
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
    configurationManager.setKernelConfigurations(id, kernelManager->getKernelCompositionConfigurations(id), composition.getParameters());
    size_t configurationsCount = configurationManager.getConfigurationCount(id);
    std::vector<KernelResult> results;

    for (size_t i = 0; i < configurationsCount; i++)
    {
        std::stringstream stream;
        stream << "Launching configuration: " << i + 1 << " / " << configurationsCount;
        logger->log(stream.str());

        KernelConfiguration currentConfiguration = configurationManager.getCurrentConfiguration(id);
        KernelResult result = kernelRunner->runComposition(id, currentConfiguration.getParameterPairs(), std::vector<ArgumentOutputDescriptor>{});

        configurationManager.calculateNextConfiguration(id, currentConfiguration, static_cast<double>(result.getComputationDuration()));
        if (validateResult(compatibilityKernel, result))
        {
            results.push_back(result);
        }
        else
        {
            results.emplace_back(composition.getName(), currentConfiguration, "Results differ");
        }

        kernelRunner->clearBuffers();
    }

    resultValidator->clearReferenceResults();
    configurationManager.clearSearcher(id);
    return results;
}

KernelResult TuningRunner::tuneKernelByStep(const KernelId id, const std::vector<ArgumentOutputDescriptor>& output)
{
    if (!kernelManager->isKernel(id))
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }

    const Kernel& kernel = kernelManager->getKernel(id);
    resultValidator->computeReferenceResult(kernel);

    if (!configurationManager.hasKernelConfigurations(id))
    {
        configurationManager.setKernelConfigurations(id, kernelManager->getKernelConfigurations(id), kernel.getParameters());
    }

    KernelConfiguration currentConfiguration = configurationManager.getCurrentConfiguration(id);
    KernelResult result = kernelRunner->runKernel(id, currentConfiguration.getParameterPairs(), output);

    configurationManager.calculateNextConfiguration(id, currentConfiguration, static_cast<double>(result.getComputationDuration()));
    validateResult(kernel, result);

    kernelRunner->clearBuffers();
    resultValidator->clearReferenceResults();
    return result;
}

KernelResult TuningRunner::tuneCompositionByStep(const KernelId id, const std::vector<ArgumentOutputDescriptor>& output)
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
        configurationManager.setKernelConfigurations(id, kernelManager->getKernelCompositionConfigurations(id), composition.getParameters());
    }

    KernelConfiguration currentConfiguration = configurationManager.getCurrentConfiguration(id);
    KernelResult result = kernelRunner->runComposition(id, currentConfiguration.getParameterPairs(), output);

    configurationManager.calculateNextConfiguration(id, currentConfiguration, static_cast<double>(result.getComputationDuration()));
    validateResult(compatibilityKernel, result);

    kernelRunner->clearBuffers();
    resultValidator->clearReferenceResults();
    return result;
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

std::vector<ParameterPair> TuningRunner::getBestConfiguration(const KernelId id) const
{
    return configurationManager.getBestConfiguration(id).getParameterPairs();
}

bool TuningRunner::validateResult(const Kernel& kernel, const KernelResult& result)
{
    if (!result.isValid())
    {
        return false;
    }

    bool resultIsCorrect = resultValidator->validateArgumentsWithClass(kernel);
    resultIsCorrect &= resultValidator->validateArgumentsWithKernel(kernel);

    if (resultIsCorrect)
    {
        logger->log(std::string("Kernel run completed successfully in ") + std::to_string((result.getComputationDuration()) / 1'000'000)
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
