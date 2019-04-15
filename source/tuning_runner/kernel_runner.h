#pragma once

#include <map>
#include <memory>
#include <utility>
#include <api/tuning_manipulator.h>
#include <compute_engine/compute_engine.h>
#include <dto/kernel_result.h>
#include <enum/kernel_run_mode.h>
#include <enum/time_unit.h>
#include <kernel/kernel_manager.h>
#include <kernel_argument/argument_manager.h>
#include <tuning_runner/manipulator_interface_implementation.h>
#include <tuning_runner/result_validator.h>

namespace ktt
{

class KernelRunner
{
public:
    // Constructor
    explicit KernelRunner(ArgumentManager* argumentManager, KernelManager* kernelManager, ComputeEngine* computeEngine);

    // Core methods
    KernelResult runKernel(const KernelId id, const KernelRunMode mode, const KernelConfiguration& configuration,
        const std::vector<OutputDescriptor>& output);
    KernelResult runKernel(const KernelId id, const KernelRunMode mode, const std::vector<ParameterPair>& configuration,
        const std::vector<OutputDescriptor>& output);
    KernelResult runComposition(const KernelId id, const KernelRunMode mode, const KernelConfiguration& configuration,
        const std::vector<OutputDescriptor>& output);
    KernelResult runComposition(const KernelId id, const KernelRunMode mode, const std::vector<ParameterPair>& configuration,
        const std::vector<OutputDescriptor>& output);
    void setTuningManipulator(const KernelId id, std::unique_ptr<TuningManipulator> manipulator);
    void setTuningManipulatorSynchronization(const KernelId id, const bool flag);
    void setTimeUnit(const TimeUnit unit);
    void setKernelProfiling(const bool flag);
    bool getKernelProfiling();

    // Result validation methods
    void setValidationMethod(const ValidationMethod method, const double toleranceThreshold);
    void setValidationMode(const ValidationMode mode);
    void setValidationRange(const ArgumentId id, const size_t range);
    void setArgumentComparator(const ArgumentId id, const std::function<bool(const void*, const void*)>& comparator);
    void setReferenceKernel(const KernelId id, const KernelId referenceId, const std::vector<ParameterPair>& referenceConfiguration,
        const std::vector<ArgumentId>& validatedArgumentIds);
    void setReferenceClass(const KernelId id, std::unique_ptr<ReferenceClass> referenceClass, const std::vector<ArgumentId>& validatedArgumentIds);
    void clearReferenceResult(const KernelId id);

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
    ResultValidator resultValidator;
    std::unique_ptr<ManipulatorInterfaceImplementation> manipulatorInterfaceImplementation;
    std::map<KernelId, std::unique_ptr<TuningManipulator>> tuningManipulators;
    std::set<KernelId> disabledSynchronizationManipulators;
    TimeUnit timeUnit;
    bool kernelProfilingFlag;

    // Helper methods
    KernelResult runKernelSimple(const Kernel& kernel, const KernelRunMode mode, const KernelConfiguration& configuration,
        const std::vector<OutputDescriptor>& output);
    KernelResult runSimpleKernelProfiling(const Kernel& kernel, const KernelRunMode mode, const KernelRuntimeData& kernelData,
        const std::vector<OutputDescriptor>& output);
    KernelResult runKernelWithManipulator(const Kernel& kernel, const KernelRunMode mode, TuningManipulator* manipulator,
        const KernelConfiguration& configuration, const std::vector<OutputDescriptor>& output);
    uint64_t runManipulatorKernelProfiling(const Kernel& kernel, const KernelRunMode mode, TuningManipulator* manipulator,
        const KernelRuntimeData& kernelData, const std::vector<OutputDescriptor>& output);
    KernelResult runCompositionWithManipulator(const KernelComposition& composition, const KernelRunMode mode, TuningManipulator* manipulator,
        const KernelConfiguration& configuration, const std::vector<OutputDescriptor>& output);
    uint64_t runCompositionProfiling(const KernelComposition& composition, const KernelRunMode mode, TuningManipulator* manipulator,
        const std::vector<KernelRuntimeData>& compositionData, const std::vector<OutputDescriptor>& output);
    uint64_t launchManipulator(const KernelId kernelId, TuningManipulator* manipulator);
    uint64_t getRemainingKernelProfilingRunsForComposition(const KernelComposition& composition,
        const std::vector<KernelRuntimeData>& compositionData);
    void validateResult(const Kernel& kernel, KernelResult& result, const KernelRunMode mode);
};

} // namespace ktt
