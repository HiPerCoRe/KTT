#pragma once

#include <map>
#include <memory>
#include <utility>
#include "manipulator_interface_implementation.h"
#include "api/tuning_manipulator.h"
#include "compute_engine/compute_engine.h"
#include "dto/kernel_result.h"
#include "kernel/kernel_manager.h"
#include "kernel_argument/argument_manager.h"
#include "utility/logger.h"

namespace ktt
{

class KernelRunner
{
public:
    // Constructor
    explicit KernelRunner(ArgumentManager* argumentManager, KernelManager* kernelManager, Logger* logger, ComputeEngine* computeEngine);

    // Core methods
    KernelResult runKernel(const KernelId id, const KernelConfiguration& configuration, const std::vector<ArgumentOutputDescriptor>& output);
    KernelResult runKernel(const KernelId id, const std::vector<ParameterPair>& configuration, const std::vector<ArgumentOutputDescriptor>& output);
    KernelResult runComposition(const KernelId id, const KernelConfiguration& configuration, const std::vector<ArgumentOutputDescriptor>& output);
    KernelResult runComposition(const KernelId id, const std::vector<ParameterPair>& configuration,
        const std::vector<ArgumentOutputDescriptor>& output);
    void setTuningManipulator(const KernelId id, std::unique_ptr<TuningManipulator> manipulator);

    // Compute engine methods
    KernelArgument downloadArgument(const ArgumentId id) const;
    void clearBuffers(const ArgumentAccessType& accessType);
    void clearBuffers();

private:
    // Attributes
    ArgumentManager* argumentManager;
    KernelManager* kernelManager;
    Logger* logger;
    ComputeEngine* computeEngine;
    std::unique_ptr<ManipulatorInterfaceImplementation> manipulatorInterfaceImplementation;
    std::map<KernelId, std::unique_ptr<TuningManipulator>> tuningManipulators;

    // Helper methods
    KernelResult runKernelSimple(const Kernel& kernel, const KernelConfiguration& configuration,
        const std::vector<ArgumentOutputDescriptor>& output);
    KernelResult runKernelWithManipulator(const Kernel& kernel, TuningManipulator* manipulator, const KernelConfiguration& configuration,
        const std::vector<ArgumentOutputDescriptor>& output);
    KernelResult runCompositionWithManipulator(const KernelComposition& composition, TuningManipulator* manipulator,
        const KernelConfiguration& configuration, const std::vector<ArgumentOutputDescriptor>& output);
};

} // namespace ktt
