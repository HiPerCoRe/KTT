#pragma once

#include <map>
#include <memory>
#include <utility>
#include <api/tuning_manipulator.h>
#include <compute_engine/compute_engine.h>
#include <dto/kernel_result.h>
#include <kernel/kernel_manager.h>
#include <kernel_argument/argument_manager.h>
#include <tuning_runner/manipulator_interface_implementation.h>

namespace ktt
{

class KernelRunner
{
public:
    // Constructor
    explicit KernelRunner(ArgumentManager* argumentManager, KernelManager* kernelManager, ComputeEngine* computeEngine);

    // Core methods
    KernelResult runKernel(const KernelId id, const KernelConfiguration& configuration, const std::vector<OutputDescriptor>& output);
    KernelResult runKernel(const KernelId id, const std::vector<ParameterPair>& configuration, const std::vector<OutputDescriptor>& output);
    KernelResult runComposition(const KernelId id, const KernelConfiguration& configuration, const std::vector<OutputDescriptor>& output);
    KernelResult runComposition(const KernelId id, const std::vector<ParameterPair>& configuration, const std::vector<OutputDescriptor>& output);
    void setTuningManipulator(const KernelId id, std::unique_ptr<TuningManipulator> manipulator);
    void setTuningManipulatorSynchronization(const KernelId id, const bool flag);
    void setKernelProfiling(const bool flag);

    // Compute engine methods
    KernelArgument downloadArgument(const ArgumentId id) const;
    void clearBuffers(const ArgumentAccessType accessType);
    void clearBuffers();
    void setPersistentArgumentUsage(const bool flag);

private:
    // Attributes
    ArgumentManager* argumentManager;
    KernelManager* kernelManager;
    ComputeEngine* computeEngine;
    std::unique_ptr<ManipulatorInterfaceImplementation> manipulatorInterfaceImplementation;
    std::map<KernelId, std::unique_ptr<TuningManipulator>> tuningManipulators;
    std::set<KernelId> disabledSynchronizationManipulators;
    bool kernelProfilingFlag;

    // Helper methods
    KernelResult runKernelSimple(const Kernel& kernel, const KernelConfiguration& configuration, const std::vector<OutputDescriptor>& output);
    KernelResult runKernelWithManipulator(const Kernel& kernel, TuningManipulator* manipulator, const KernelConfiguration& configuration,
        const std::vector<OutputDescriptor>& output);
    uint64_t launchManipulator(const KernelId kernelId, TuningManipulator* manipulator);
    KernelResult runCompositionWithManipulator(const KernelComposition& composition, TuningManipulator* manipulator,
        const KernelConfiguration& configuration, const std::vector<OutputDescriptor>& output);
};

} // namespace ktt
