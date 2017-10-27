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
    std::vector<TuningResult> tuneKernel(const KernelId id);
    std::vector<TuningResult> tuneKernelComposition(const KernelId id);
    void runKernel(const KernelId id, const std::vector<ParameterPair>& configuration, const std::vector<ArgumentOutputDescriptor>& output);
    void runComposition(const KernelId id, const std::vector<ParameterPair>& configuration, const std::vector<ArgumentOutputDescriptor>& output);
    void setSearchMethod(const SearchMethod& method, const std::vector<double>& arguments);
    void setValidationMethod(const ValidationMethod& method, const double toleranceThreshold);
    void setValidationRange(const ArgumentId id, const size_t range);
    void setReferenceKernel(const KernelId id, const KernelId referenceId, const std::vector<ParameterPair>& referenceConfiguration,
        const std::vector<ArgumentId>& validatedArgumentIds);
    void setReferenceClass(const KernelId id, std::unique_ptr<ReferenceClass> referenceClass, const std::vector<ArgumentId>& validatedArgumentIds);
    void setTuningManipulator(const KernelId id, std::unique_ptr<TuningManipulator> manipulator);

private:
    // Attributes
    ArgumentManager* argumentManager;
    KernelManager* kernelManager;
    Logger* logger;
    ComputeEngine* computeEngine;
    std::unique_ptr<ResultValidator> resultValidator;
    std::map<KernelId, std::unique_ptr<TuningManipulator>> tuningManipulators;
    std::unique_ptr<ManipulatorInterfaceImplementation> manipulatorInterfaceImplementation;
    SearchMethod searchMethod;
    std::vector<double> searchArguments;
    RunMode runMode;

    // Helper methods
    TuningResult runKernelSimple(const Kernel& kernel, const KernelConfiguration& configuration,
        const std::vector<ArgumentOutputDescriptor>& output);
    TuningResult runKernelWithManipulator(const Kernel& kernel, TuningManipulator* manipulator, const KernelConfiguration& configuration,
        const std::vector<ArgumentOutputDescriptor>& output);
    TuningResult runCompositionWithManipulator(const KernelComposition& composition, TuningManipulator* manipulator,
        const KernelConfiguration& configuration, const std::vector<ArgumentOutputDescriptor>& output);
    std::unique_ptr<Searcher> getSearcher(const SearchMethod& method, const std::vector<double>& arguments,
        const std::vector<KernelConfiguration>& configurations, const std::vector<KernelParameter>& parameters) const;
    bool validateResult(const Kernel& kernel, const TuningResult& result);
    std::string getSearchMethodName(const SearchMethod& method) const;
    Kernel compositionToKernel(const KernelComposition& composition) const;
};

} // namespace ktt
