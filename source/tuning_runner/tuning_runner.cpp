#include <algorithm>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include "tuning_runner.h"
#include "utility/ktt_utility.h"
#include "utility/logger.h"
#include "utility/timer.h"
#include "utility/result_loader.h"

namespace ktt
{

TuningRunner::TuningRunner(ArgumentManager* argumentManager, KernelManager* kernelManager, KernelRunner* kernelRunner) :
    argumentManager(argumentManager),
    kernelManager(kernelManager),
    kernelRunner(kernelRunner),
    resultValidator(std::make_unique<ResultValidator>(argumentManager, kernelRunner))
{}

std::vector<KernelResult> TuningRunner::tuneKernel(const KernelId id, std::unique_ptr<StopCondition> stopCondition)
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
    if (!configurationManager.hasKernelConfigurations(id))
    {
        configurationManager.setKernelConfigurations(id, kernelManager->getKernelConfigurations(id));
    }

    size_t configurationCount = configurationManager.getConfigurationCount(id);
    if (stopCondition != nullptr)
    {
        stopCondition->initialize(configurationCount);
        configurationCount = std::min(configurationCount, stopCondition->getConfigurationCount());
    }

    std::vector<KernelResult> results;

    for (size_t i = 0; i < configurationCount; i++)
    {
        std::stringstream stream;
        stream << "Launching configuration " << i + 1 << "/" << configurationCount << " for kernel " << kernel.getName();
        Logger::getLogger().log(LoggingLevel::Info, stream.str());

        KernelConfiguration currentConfiguration = configurationManager.getCurrentConfiguration(id);
        KernelResult result = kernelRunner->runKernel(id, currentConfiguration, std::vector<OutputDescriptor>{});

        if (validateResult(kernel, result))
        {
            results.push_back(result);
            configurationManager.calculateNextConfiguration(id, kernel.getName(), currentConfiguration, result.getComputationDuration());

            if (stopCondition != nullptr)
            {
                stopCondition->updateStatus(static_cast<double>(result.getComputationDuration()));
            }
        }
        else
        {
            results.emplace_back(kernel.getName(), currentConfiguration, "Results differ");
            configurationManager.calculateNextConfiguration(id, kernel.getName(), currentConfiguration, UINT64_MAX);

            if (stopCondition != nullptr)
            {
                stopCondition->updateStatus(std::numeric_limits<double>::max());
            }
        }

        kernelRunner->clearBuffers(ArgumentAccessType::ReadWrite);
        kernelRunner->clearBuffers(ArgumentAccessType::WriteOnly);
        if (kernel.hasTuningManipulator())
        {
            kernelRunner->clearBuffers(ArgumentAccessType::ReadOnly);
        }

        if (stopCondition != nullptr && stopCondition->isMet())
        {
            Logger::getLogger().log(LoggingLevel::Info, stopCondition->getStatusString());
            break;
        }
    }

    kernelRunner->clearBuffers();
    resultValidator->clearReferenceResults();
    configurationManager.clearKernelData(id, false, false);
    return results;
}

std::vector<KernelResult> TuningRunner::dryTuneKernel(const KernelId id, const std::string& filePath, const size_t iterations)
{
    if (!kernelManager->isKernel(id))
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }

    ResultLoader resultLoader;
    if (!resultLoader.loadResults(filePath))
    {
        throw std::runtime_error(std::string("Unable to open file: ") + filePath);
    }

    const Kernel& kernel = kernelManager->getKernel(id);
    if (!configurationManager.hasKernelConfigurations(id))
    {
        configurationManager.setKernelConfigurations(id, kernelManager->getKernelConfigurations(id));
    }

    size_t configurationCount = configurationManager.getConfigurationCount(id);
    std::vector<KernelResult> results;
    size_t tuningIterations;
    if (iterations == 0)
    {
        tuningIterations = configurationCount;
    }
    else
    {
        tuningIterations = std::min(configurationCount, iterations);
    }

    for (size_t i = 0; i < tuningIterations; i++)
    {
        KernelConfiguration currentConfiguration = configurationManager.getCurrentConfiguration(id);
        KernelResult result(kernel.getName(), currentConfiguration);

        try
        {
            std::stringstream stream;
            stream << "Launching configuration " << i + 1 << "/" << configurationCount << " for kernel " << kernel.getName() << ": "
                << currentConfiguration;
            Logger::getLogger().log(LoggingLevel::Info, stream.str());

            result = resultLoader.readResult(currentConfiguration);
            result.setConfiguration(currentConfiguration);
        }
        catch (const std::runtime_error& error)
        {
            Logger::getLogger().log(LoggingLevel::Warning, std::string("Kernel run failed, reason: ") + error.what());
            results.emplace_back(kernel.getName(), currentConfiguration, std::string("Failed kernel run: ") + error.what());
        }

        configurationManager.calculateNextConfiguration(id, kernel.getName(), currentConfiguration, result.getComputationDuration());
        results.push_back(result);
    }

    configurationManager.clearKernelData(id, false, false);
    return results;
}

std::vector<KernelResult> TuningRunner::tuneComposition(const KernelId id, std::unique_ptr<StopCondition> stopCondition)
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
    if (!configurationManager.hasKernelConfigurations(id))
    {
        configurationManager.setKernelConfigurations(id, kernelManager->getKernelCompositionConfigurations(id));
    }

    size_t configurationCount = configurationManager.getConfigurationCount(id);
    if (stopCondition != nullptr)
    {
        stopCondition->initialize(configurationCount);
        configurationCount = std::min(configurationCount, stopCondition->getConfigurationCount());
    }

    std::vector<KernelResult> results;

    for (size_t i = 0; i < configurationCount; i++)
    {
        std::stringstream stream;
        stream << "Launching configuration " << i + 1 << "/" << configurationCount << " for kernel composition " << composition.getName();
        Logger::getLogger().log(LoggingLevel::Info, stream.str());

        KernelConfiguration currentConfiguration = configurationManager.getCurrentConfiguration(id);
        KernelResult result = kernelRunner->runComposition(id, currentConfiguration, std::vector<OutputDescriptor>{});

        if (validateResult(compatibilityKernel, result))
        {
            results.push_back(result);
            configurationManager.calculateNextConfiguration(id, composition.getName(), currentConfiguration, result.getComputationDuration());

            if (stopCondition != nullptr)
            {
                stopCondition->updateStatus(static_cast<double>(result.getComputationDuration()));
            }
        }
        else
        {
            results.emplace_back(composition.getName(), currentConfiguration, "Results differ");
            configurationManager.calculateNextConfiguration(id, composition.getName(), currentConfiguration, UINT64_MAX);

            if (stopCondition != nullptr)
            {
                stopCondition->updateStatus(std::numeric_limits<double>::max());
            }
        }

        kernelRunner->clearBuffers();

        if (stopCondition != nullptr && stopCondition->isMet())
        {
            Logger::getLogger().log(LoggingLevel::Info, stopCondition->getStatusString());
            break;
        }
    }

    resultValidator->clearReferenceResults();
    configurationManager.clearKernelData(id, false, false);
    return results;
}

KernelResult TuningRunner::tuneKernelByStep(const KernelId id, const std::vector<OutputDescriptor>& output, const bool recomputeReference)
{
    if (!kernelManager->isKernel(id))
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }

    const Kernel& kernel = kernelManager->getKernel(id);
    if (recomputeReference)
    {
        resultValidator->clearReferenceResults(id);
        resultValidator->computeReferenceResult(kernel);
    }

    if (!configurationManager.hasKernelConfigurations(id))
    {
        configurationManager.setKernelConfigurations(id, kernelManager->getKernelConfigurations(id));
    }

    KernelConfiguration currentConfiguration = configurationManager.getCurrentConfiguration(id);
    KernelResult result = kernelRunner->runKernel(id, currentConfiguration, output);

    if (validateResult(kernel, result))
    {
        configurationManager.calculateNextConfiguration(id, kernel.getName(), currentConfiguration, result.getComputationDuration());
    }
    else
    {
        result.setValid(false);
        result.setErrorMessage("Results differ");
        configurationManager.calculateNextConfiguration(id, kernel.getName(), currentConfiguration, UINT64_MAX);
    }

    kernelRunner->clearBuffers();
    return result;
}

KernelResult TuningRunner::tuneCompositionByStep(const KernelId id, const std::vector<OutputDescriptor>& output, const bool recomputeReference)
{
    if (!kernelManager->isComposition(id))
    {
        throw std::runtime_error(std::string("Invalid kernel composition id: ") + std::to_string(id));
    }

    const KernelComposition& composition = kernelManager->getKernelComposition(id);
    const Kernel compatibilityKernel = composition.transformToKernel();
    if (recomputeReference)
    {
        resultValidator->clearReferenceResults(id);
        resultValidator->computeReferenceResult(compatibilityKernel);
    }

    if (!configurationManager.hasKernelConfigurations(id))
    {
        configurationManager.setKernelConfigurations(id, kernelManager->getKernelCompositionConfigurations(id));
    }

    KernelConfiguration currentConfiguration = configurationManager.getCurrentConfiguration(id);
    KernelResult result = kernelRunner->runComposition(id, currentConfiguration, output);

    if (validateResult(compatibilityKernel, result))
    {
        configurationManager.calculateNextConfiguration(id, composition.getName(), currentConfiguration, result.getComputationDuration());
    }
    else
    {
        result.setValid(false);
        result.setErrorMessage("Results differ");
        configurationManager.calculateNextConfiguration(id, composition.getName(), currentConfiguration, UINT64_MAX);
    }

    kernelRunner->clearBuffers();
    return result;
}

void TuningRunner::clearKernelData(const KernelId id, const bool clearConfigurations)
{
    configurationManager.clearKernelData(id, clearConfigurations, true);
}

void TuningRunner::setSearchMethod(const SearchMethod method, const std::vector<double>& arguments)
{
    configurationManager.setSearchMethod(method, arguments);
}

void TuningRunner::setValidationMethod(const ValidationMethod method, const double toleranceThreshold)
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

ComputationResult TuningRunner::getBestComputationResult(const KernelId id) const
{
    return configurationManager.getBestComputationResult(id);
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
        Logger::getLogger().log(LoggingLevel::Info, std::string("Kernel run completed successfully in ")
            + std::to_string((result.getComputationDuration()) / 1'000'000) + "ms");
    }
    else
    {
        Logger::getLogger().log(LoggingLevel::Warning, "Kernel run completed successfully, but results differ");
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
