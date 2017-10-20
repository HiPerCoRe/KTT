#pragma once

#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "manipulator_interface_implementation.h"
#include "result_validator.h"
#include "searcher/searcher.h"
#include "api/tuning_manipulator.h"
#include "compute_engine/compute_engine.h"
#include "dto/tuning_result.h"
#include "enum/search_method.h"
#include "kernel/kernel_manager.h"
#include "kernel_argument/argument_manager.h"
#include "utility/logger.h"

namespace ktt
{

class TuningRunner
{
public:
    // Constructor
    explicit TuningRunner(ArgumentManager* argumentManager, KernelManager* kernelManager, Logger* logger, ComputeEngine* computeEngine,
        const RunMode& runMode);

    // Core methods
    std::vector<TuningResult> tuneKernel(const size_t id);
    std::vector<TuningResult> tuneKernelComposition(const size_t id);
    void runKernel(const size_t kernelId, const std::vector<ParameterValue>& kernelConfiguration,
        const std::vector<ArgumentOutputDescriptor>& outputDescriptors);
    void runKernelComposition(const size_t compositionId, const std::vector<ParameterValue>& compositionConfiguration,
        const std::vector<ArgumentOutputDescriptor>& outputDescriptors);
    void setSearchMethod(const SearchMethod& searchMethod, const std::vector<double>& searchArguments);
    void setValidationMethod(const ValidationMethod& validationMethod, const double toleranceThreshold);
    void setValidationRange(const size_t argumentId, const size_t validationRange);
    void setReferenceKernel(const size_t kernelId, const size_t referenceKernelId, const std::vector<ParameterValue>& referenceKernelConfiguration,
        const std::vector<size_t>& resultArgumentIds);
    void setReferenceClass(const size_t kernelId, std::unique_ptr<ReferenceClass> referenceClass, const std::vector<size_t>& resultArgumentIds);
    void setTuningManipulator(const size_t kernelId, std::unique_ptr<TuningManipulator> tuningManipulator);

private:
    // Attributes
    ArgumentManager* argumentManager;
    KernelManager* kernelManager;
    Logger* logger;
    ComputeEngine* computeEngine;
    std::unique_ptr<ResultValidator> resultValidator;
    std::map<size_t, std::unique_ptr<TuningManipulator>> manipulatorMap;
    std::unique_ptr<ManipulatorInterfaceImplementation> manipulatorInterfaceImplementation;
    SearchMethod searchMethod;
    std::vector<double> searchArguments;
    RunMode runMode;

    // Helper methods
    TuningResult runKernelSimple(const Kernel& kernel, const KernelConfiguration& currentConfiguration,
        const std::vector<ArgumentOutputDescriptor>& outputDescriptors);
    TuningResult runKernelWithManipulator(const Kernel& kernel, TuningManipulator* manipulator, const KernelConfiguration& currentConfiguration,
        const std::vector<ArgumentOutputDescriptor>& outputDescriptors);
    TuningResult runKernelCompositionWithManipulator(const KernelComposition& kernelComposition, TuningManipulator* manipulator,
        const KernelConfiguration& currentConfiguration, const std::vector<ArgumentOutputDescriptor>& outputDescriptors);
    std::unique_ptr<Searcher> getSearcher(const SearchMethod& searchMethod, const std::vector<double>& searchArguments,
        const std::vector<KernelConfiguration>& configurations, const std::vector<KernelParameter>& parameters) const;
    bool validateResult(const Kernel& kernel, const TuningResult& tuningResult);
    std::string getSearchMethodName(const SearchMethod& searchMethod) const;
    Kernel compositionToKernel(const KernelComposition& composition) const;
};

} // namespace ktt
