#include <fstream>
#include <iterator>
#include <stdexcept>
#include <sstream>
#include <string>
#include "kernel_runner.h"
#include "utility/ktt_utility.h"
#include "utility/logger.h"
#include "utility/timer.h"

namespace ktt
{

KernelRunner::KernelRunner(ArgumentManager* argumentManager, KernelManager* kernelManager, ComputeEngine* computeEngine) :
    argumentManager(argumentManager),
    kernelManager(kernelManager),
    computeEngine(computeEngine),
    manipulatorInterfaceImplementation(std::make_unique<ManipulatorInterfaceImplementation>(computeEngine))
{}

KernelResult KernelRunner::runKernel(const KernelId id, const KernelConfiguration& configuration, const std::vector<OutputDescriptor>& output)
{
    if (!kernelManager->isKernel(id))
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }

    const Kernel& kernel = kernelManager->getKernel(id);

    std::stringstream stream;
    stream << "Running kernel " << kernel.getName() << " with configuration: " << configuration;
    Logger::getLogger().log(LoggingLevel::Info, stream.str());

    KernelResult result;
    try
    {
        if (kernel.hasTuningManipulator())
        {
            auto manipulatorPointer = tuningManipulators.find(id);
            result = runKernelWithManipulator(kernel, manipulatorPointer->second.get(), configuration, output);
        }
        else
        {
            result = runKernelSimple(kernel, configuration, output);
        }
    }
    catch (const std::runtime_error& error)
    {
        computeEngine->synchronizeDevice();
        computeEngine->clearEvents();
        Logger::getLogger().log(LoggingLevel::Warning, std::string("Kernel run failed, reason: ") + error.what());
        result = KernelResult(kernel.getName(), configuration, error.what());
    }

    return result;
}

KernelResult KernelRunner::runKernel(const KernelId id, const std::vector<ParameterPair>& configuration, const std::vector<OutputDescriptor>& output)
{
    const KernelConfiguration launchConfiguration = kernelManager->getKernelConfiguration(id, configuration);
    return runKernel(id, launchConfiguration, output);
}

KernelResult KernelRunner::runComposition(const KernelId id, const KernelConfiguration& configuration, const std::vector<OutputDescriptor>& output)
{
    if (!kernelManager->isComposition(id))
    {
        throw std::runtime_error(std::string("Invalid kernel composition id: ") + std::to_string(id));
    }

    const KernelComposition& composition = kernelManager->getKernelComposition(id);

    std::stringstream stream;
    stream << "Running kernel composition " << composition.getName() << " with configuration: " << configuration;
    Logger::getLogger().log(LoggingLevel::Info, stream.str());

    KernelResult result;
    try
    {
        auto manipulatorPointer = tuningManipulators.find(id);
        result = runCompositionWithManipulator(composition, manipulatorPointer->second.get(), configuration, output);
    }
    catch (const std::runtime_error& error)
    {
        computeEngine->synchronizeDevice();
        computeEngine->clearEvents();
        Logger::getLogger().log(LoggingLevel::Warning, std::string("Kernel composition run failed, reason: ") + error.what());
        result = KernelResult(composition.getName(), configuration, error.what());
    }

    return result;
}

KernelResult KernelRunner::runComposition(const KernelId id, const std::vector<ParameterPair>& configuration,
    const std::vector<OutputDescriptor>& output)
{
    const KernelConfiguration launchConfiguration = kernelManager->getKernelCompositionConfiguration(id, configuration);
    return runComposition(id, launchConfiguration, output);
}

void KernelRunner::setTuningManipulator(const KernelId id, std::unique_ptr<TuningManipulator> manipulator)
{
    if (tuningManipulators.find(id) != tuningManipulators.end())
    {
        tuningManipulators.erase(id);
    }
    tuningManipulators.insert(std::make_pair(id, std::move(manipulator)));
}

void KernelRunner::setTuningManipulatorSynchronization(const KernelId id, const bool flag)
{
    if (flag && disabledSynchronizationManipulators.find(id) != disabledSynchronizationManipulators.end())
    {
        disabledSynchronizationManipulators.erase(id);
    }
    else if (!flag && disabledSynchronizationManipulators.find(id) == disabledSynchronizationManipulators.end())
    {
        disabledSynchronizationManipulators.insert(id);
    }
}

KernelArgument KernelRunner::downloadArgument(const ArgumentId id) const
{
    return computeEngine->downloadArgumentObject(id, nullptr);
}

void KernelRunner::clearBuffers(const ArgumentAccessType accessType)
{
    computeEngine->clearBuffers(accessType);
}

void KernelRunner::clearBuffers()
{
    computeEngine->clearBuffers();
}

void KernelRunner::setPersistentArgumentUsage(const bool flag)
{
    computeEngine->setPersistentBufferUsage(flag);
}

KernelResult KernelRunner::runKernelSimple(const Kernel& kernel, const KernelConfiguration& configuration,
    const std::vector<OutputDescriptor>& output)
{
    KernelId kernelId = kernel.getId();
    std::string kernelName = kernel.getName();
    std::string source = kernelManager->getKernelSourceWithDefines(kernelId, configuration);

    KernelRuntimeData kernelData(kernelId, kernelName, source, configuration.getGlobalSize(), configuration.getLocalSize(), kernel.getArgumentIds(),
        configuration.getLocalMemoryModifiers());
    KernelResult result = computeEngine->runKernel(kernelData, argumentManager->getArguments(kernel.getArgumentIds()), output);
    result.setConfiguration(configuration);
    return result;
}

KernelResult KernelRunner::runKernelWithManipulator(const Kernel& kernel, TuningManipulator* manipulator, const KernelConfiguration& configuration,
    const std::vector<OutputDescriptor>& output)
{
    KernelId kernelId = kernel.getId();
    std::string source = kernelManager->getKernelSourceWithDefines(kernelId, configuration);
    KernelRuntimeData kernelData(kernelId, kernel.getName(), source, configuration.getGlobalSize(), configuration.getLocalSize(),
        kernel.getArgumentIds(), configuration.getLocalMemoryModifiers());

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
        if (disabledSynchronizationManipulators.find(kernel.getId()) == disabledSynchronizationManipulators.end())
        {
            manipulatorInterfaceImplementation->synchronizeDeviceInternal();
        }
        timer.stop();
    }
    catch (const std::runtime_error&)
    {
        manipulatorInterfaceImplementation->synchronizeDeviceInternal();
        manipulatorInterfaceImplementation->clearData();
        manipulator->manipulatorInterface = nullptr;
        throw;
    }

    manipulatorInterfaceImplementation->downloadBuffers(output);
    KernelResult result = manipulatorInterfaceImplementation->getCurrentResult();
    manipulatorInterfaceImplementation->clearData();
    manipulator->manipulatorInterface = nullptr;

    size_t manipulatorDuration = timer.getElapsedTime();
    manipulatorDuration -= result.getOverhead();
    result.setKernelName(kernel.getName());
    result.setComputationDuration(manipulatorDuration);
    return result;
}

KernelResult KernelRunner::runCompositionWithManipulator(const KernelComposition& composition, TuningManipulator* manipulator,
    const KernelConfiguration& configuration, const std::vector<OutputDescriptor>& output)
{
    manipulator->manipulatorInterface = manipulatorInterfaceImplementation.get();
    std::vector<KernelArgument*> allArguments = argumentManager->getArguments(composition.getSharedArgumentIds());

    for (const auto kernel : composition.getKernels())
    {
        KernelId kernelId = kernel->getId();
        std::vector<ArgumentId> argumentIds = composition.getKernelArgumentIds(kernelId);
        std::string source = kernelManager->getKernelSourceWithDefines(kernelId, configuration);

        KernelRuntimeData kernelData(kernelId, kernel->getName(), source, configuration.getCompositionKernelGlobalSize(kernelId),
            configuration.getCompositionKernelLocalSize(kernelId), argumentIds, configuration.getCompositionKernelLocalMemoryModifiers(kernelId));
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
        if (disabledSynchronizationManipulators.find(composition.getId()) == disabledSynchronizationManipulators.end())
        {
            manipulatorInterfaceImplementation->synchronizeDeviceInternal();
        }
        timer.stop();
    }
    catch (const std::runtime_error&)
    {
        manipulatorInterfaceImplementation->synchronizeDeviceInternal();
        manipulatorInterfaceImplementation->clearData();
        manipulator->manipulatorInterface = nullptr;
        throw;
    }

    manipulatorInterfaceImplementation->downloadBuffers(output);
    KernelResult result = manipulatorInterfaceImplementation->getCurrentResult();
    manipulatorInterfaceImplementation->clearData();
    manipulator->manipulatorInterface = nullptr;

    size_t manipulatorDuration = timer.getElapsedTime();
    manipulatorDuration -= result.getOverhead();
    result.setKernelName(composition.getName());
    result.setComputationDuration(manipulatorDuration);
    return result;
}

} // namespace ktt
