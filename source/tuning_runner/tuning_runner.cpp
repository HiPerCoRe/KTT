#include <algorithm>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <tuning_runner/tuning_runner.h>
#include <utility/ktt_utility.h>
#include <utility/logger.h>
#include <utility/result_loader.h>
#include <utility/timer.h>

namespace ktt
{

TuningRunner::TuningRunner(ArgumentManager* argumentManager, KernelManager* kernelManager, KernelRunner* kernelRunner, const DeviceInfo& info) :
    argumentManager(argumentManager),
    kernelManager(kernelManager),
    kernelRunner(kernelRunner),
    configurationManager(info)
{}

std::vector<ComputationResult> TuningRunner::tuneKernel(const KernelId id, std::unique_ptr<StopCondition> stopCondition)
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

    if (!configurationManager.hasKernelConfigurations(id))
    {
        configurationManager.initializeConfigurations(kernel);
    }

    size_t configurationCount = configurationManager.getConfigurationCount(id);
    std::vector<ComputationResult> results;

    if (stopCondition != nullptr)
    {
        stopCondition->initialize(configurationCount);
        configurationCount = std::min(configurationCount, stopCondition->getConfigurationCount());
    }

    for (size_t i = 0; i < configurationCount; ++i)
    {
        std::stringstream stream;
        stream << "Launching configuration " << i + 1 << "/" << configurationCount << " for kernel " << kernel.getName();
        Logger::logInfo(stream.str());

        const ComputationResult result = tuneKernelByStep(id, KernelRunMode::OfflineTuning, std::vector<OutputDescriptor>{}, false);
        results.push_back(result);

        if (stopCondition != nullptr)
        {
            stopCondition->updateStatus(result);

            if (stopCondition->isSatisfied())
            {
                Logger::logInfo(stopCondition->getStatusString());
                break;
            }
        }
    }

    kernelRunner->clearBuffers();
    kernelRunner->clearReferenceResult(id);
    configurationManager.clearKernelData(id, false, false);
    return results;
}

std::vector<ComputationResult> TuningRunner::dryTuneKernel(const KernelId id, const std::string& filePath, const size_t iterations)
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
        configurationManager.initializeConfigurations(kernel);
    }

    size_t configurationCount = configurationManager.getConfigurationCount(id);
    std::vector<ComputationResult> results;
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
        KernelConfiguration currentConfiguration = configurationManager.getCurrentConfiguration(kernel);
        KernelResult result(kernel.getName(), currentConfiguration);

        try
        {
            std::stringstream stream;
            stream << "Launching configuration " << i + 1 << "/" << configurationCount << " for kernel " << kernel.getName() << ": "
                << currentConfiguration;
            Logger::logInfo(stream.str());

            result = resultLoader.readResult(currentConfiguration);
            result.setConfiguration(currentConfiguration);

            if (result.isValid())
            {
                results.emplace_back(result.getKernelName(), result.getConfiguration().getParameterPairs(), result.getComputationDuration(),
                    result.getCompilationData(), result.getProfilingData());
            }
            else
            {
                results.emplace_back(result.getKernelName(), result.getConfiguration().getParameterPairs(), result.getErrorMessage());
            }
        }
        catch (const std::runtime_error& error)
        {
            const std::string errorMessage = std::string("Kernel run failed, reason: ") + error.what();
            Logger::logWarning(errorMessage);
            result.setValid(false);
            result.setErrorMessage(errorMessage);
            results.emplace_back(result.getKernelName(), result.getConfiguration().getParameterPairs(), errorMessage);
        }

        configurationManager.calculateNextConfiguration(kernel, result);
        resultPrinter.addResult(id, result);
    }

    configurationManager.clearKernelData(id, false, false);
    return results;
}

std::vector<ComputationResult> TuningRunner::tuneComposition(const KernelId id, std::unique_ptr<StopCondition> stopCondition)
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

    if (!configurationManager.hasKernelConfigurations(id))
    {
        configurationManager.initializeConfigurations(composition);
    }

    size_t configurationCount = configurationManager.getConfigurationCount(id);
    std::vector<ComputationResult> results;

    if (stopCondition != nullptr)
    {
        stopCondition->initialize(configurationCount);
        configurationCount = std::min(configurationCount, stopCondition->getConfigurationCount());
    }

    for (size_t i = 0; i < configurationCount; ++i)
    {
        std::stringstream stream;
        stream << "Launching configuration " << i + 1 << "/" << configurationCount << " for kernel composition " << composition.getName();
        Logger::logInfo(stream.str());

        const ComputationResult result = tuneCompositionByStep(id, KernelRunMode::OfflineTuning, std::vector<OutputDescriptor>{}, false);
        results.push_back(result);

        if (stopCondition != nullptr)
        {
            stopCondition->updateStatus(result);

            if (stopCondition->isSatisfied())
            {
                Logger::logInfo(stopCondition->getStatusString());
                break;
            }
        }
    }

    kernelRunner->clearBuffers();
    kernelRunner->clearReferenceResult(id);
    configurationManager.clearKernelData(id, false, false);
    return results;
}

ComputationResult TuningRunner::tuneKernelByStep(const KernelId id, const KernelRunMode mode, const std::vector<OutputDescriptor>& output,
    const bool recomputeReference)
{
    if (!kernelManager->isKernel(id))
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }

    const Kernel& kernel = kernelManager->getKernel(id);
    if (recomputeReference)
    {
        kernelRunner->clearReferenceResult(id);
    }

    if (!configurationManager.hasKernelConfigurations(id))
    {
        configurationManager.initializeConfigurations(kernel);
    }

    KernelConfiguration currentConfiguration = configurationManager.getCurrentConfiguration(kernel);
    KernelResult result = kernelRunner->runKernel(id, mode, currentConfiguration, output);

    if (!kernelRunner->getKernelProfiling() || result.getProfilingData().getRemainingProfilingRuns() == 0)
    {
        configurationManager.calculateNextConfiguration(kernel, result);
        resultPrinter.addResult(id, result);
    }

    if (kernel.hasTuningManipulator() || mode != KernelRunMode::OfflineTuning)
    {
        kernelRunner->clearBuffers(ArgumentAccessType::ReadOnly);
    }

    kernelRunner->clearBuffers(ArgumentAccessType::WriteOnly);
    kernelRunner->clearBuffers(ArgumentAccessType::ReadWrite);

    if (!result.isValid())
    {
        return ComputationResult(result.getKernelName(), result.getConfiguration().getParameterPairs(), result.getErrorMessage());
    }

    return ComputationResult(result.getKernelName(), result.getConfiguration().getParameterPairs(), result.getComputationDuration(),
        result.getCompilationData(), result.getProfilingData());
}

ComputationResult TuningRunner::tuneCompositionByStep(const KernelId id, const KernelRunMode mode, const std::vector<OutputDescriptor>& output,
    const bool recomputeReference)
{
    if (!kernelManager->isComposition(id))
    {
        throw std::runtime_error(std::string("Invalid kernel composition id: ") + std::to_string(id));
    }

    const KernelComposition& composition = kernelManager->getKernelComposition(id);
    if (recomputeReference)
    {
        kernelRunner->clearReferenceResult(id);
    }

    if (!configurationManager.hasKernelConfigurations(id))
    {
        configurationManager.initializeConfigurations(composition);
    }

    KernelConfiguration currentConfiguration = configurationManager.getCurrentConfiguration(composition);
    KernelResult result = kernelRunner->runComposition(id, mode, currentConfiguration, output);
    
    if (!kernelRunner->getKernelProfiling() || result.getProfilingData().getRemainingProfilingRuns() == 0)
    {
        configurationManager.calculateNextConfiguration(composition, result);
        resultPrinter.addResult(id, result);
    }
    
    kernelRunner->clearBuffers();

    if (!result.isValid())
    {
        return ComputationResult(result.getKernelName(), result.getConfiguration().getParameterPairs(), result.getErrorMessage());
    }

    return ComputationResult(result.getKernelName(), result.getConfiguration().getParameterPairs(), result.getComputationDuration(),
        result.getCompositionCompilationData(), result.getCompositionProfilingData());
}

void TuningRunner::clearKernelData(const KernelId id, const bool clearConfigurations)
{
    resultPrinter.clearResults(id);
    configurationManager.clearKernelData(id, clearConfigurations, true);
}

void TuningRunner::setKernelProfiling(const bool flag)
{
    kernelRunner->setKernelProfiling(flag);
}

void TuningRunner::setSearchMethod(const SearchMethod method, const std::vector<double>& arguments)
{
    configurationManager.setSearchMethod(method, arguments);
}

ComputationResult TuningRunner::getBestComputationResult(const KernelId id) const
{
    return configurationManager.getBestComputationResult(id);
}

void TuningRunner::setTimeUnit(const TimeUnit unit)
{
    resultPrinter.setTimeUnit(unit);
}

void TuningRunner::setInvalidResultPrinting(const bool flag)
{
    resultPrinter.setInvalidResultPrinting(flag);
}

void TuningRunner::printResult(const KernelId id, std::ostream& outputTarget, const PrintFormat format) const
{
    resultPrinter.printResult(id, outputTarget, format);
}

void TuningRunner::printResult(const KernelId id, const std::string& filePath, const PrintFormat format) const
{
    std::ofstream outputFile(filePath);

    if (!outputFile.is_open())
    {
        throw std::runtime_error(std::string("Unable to open file: ") + filePath);
    }

    resultPrinter.printResult(id, outputFile, format);
}

bool TuningRunner::hasWritableZeroCopyArguments(const Kernel& kernel) const
{
    std::vector<KernelArgument*> arguments = argumentManager->getArguments(kernel.getArgumentIds());

    for (const auto* argument : arguments)
    {
        if (argument->getMemoryLocation() == ArgumentMemoryLocation::HostZeroCopy && argument->getAccessType() != ArgumentAccessType::ReadOnly)
        {
            return true;
        }
    }

    return false;
}

} // namespace ktt
