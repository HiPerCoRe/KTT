#pragma once

#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "manipulator_interface_implementation.h"
#include "result_validator.h"
#include "searcher/searcher.h"
#include "api/tuning_manipulator.h"
#include "compute_api_driver/compute_api_driver.h"
#include "dto/tuning_result.h"
#include "kernel/kernel_manager.h"
#include "kernel_argument/argument_manager.h"
#include "utility/logger.h"

namespace ktt
{

class TuningRunner
{
public:
    // Constructor
    explicit TuningRunner(ArgumentManager* argumentManager, KernelManager* kernelManager, Logger* logger, ComputeApiDriver* computeApiDriver);

    // Core methods
    std::vector<TuningResult> tuneKernel(const size_t id);
    void runKernel(const size_t kernelId, const std::vector<ParameterValue>& kernelConfiguration);
    void setValidationMethod(const ValidationMethod& validationMethod, const double toleranceThreshold);
    void setValidationRange(const size_t argumentId, const size_t validationRange);
    void setReferenceKernel(const size_t kernelId, const size_t referenceKernelId, const std::vector<ParameterValue>& referenceKernelConfiguration,
        const std::vector<size_t>& resultArgumentIds);
    void setReferenceClass(const size_t kernelId, std::unique_ptr<ReferenceClass> referenceClass, const std::vector<size_t>& resultArgumentIds);
    void setTuningManipulator(const size_t kernelId, std::unique_ptr<TuningManipulator> tuningManipulator);
    void enableArgumentPrinting(const size_t argumentId, const std::string& filePath, const ArgumentPrintCondition& argumentPrintCondition);

private:
    // Attributes
    ArgumentManager* argumentManager;
    KernelManager* kernelManager;
    Logger* logger;
    ComputeApiDriver* computeApiDriver;
    ResultValidator resultValidator;
    std::map<size_t, std::unique_ptr<TuningManipulator>> manipulatorMap;
    std::unique_ptr<ManipulatorInterfaceImplementation> manipulatorInterfaceImplementation;

    // Helper methods
    std::pair<KernelRunResult, uint64_t> runKernel(const Kernel* kernel, const KernelConfiguration& currentConfiguration,
        const size_t currentConfigurationIndex, const size_t configurationsCount);
    std::pair<KernelRunResult, uint64_t> runKernelWithManipulator(TuningManipulator* manipulator,
        const std::vector<std::pair<size_t, KernelRuntimeData>>& kernelDataVector, const KernelConfiguration& currentConfiguration);
    std::unique_ptr<Searcher> getSearcher(const SearchMethod& searchMethod, const std::vector<double>& searchArguments,
        const std::vector<KernelConfiguration>& configurations, const std::vector<KernelParameter>& parameters) const;
    std::vector<KernelArgument> getKernelArguments(const size_t kernelId) const;
    std::vector<const KernelArgument*> getKernelArgumentPointers(const size_t kernelId) const;
    std::vector<std::pair<size_t, KernelRuntimeData>> getKernelDataVector(const size_t tunedKernelId, const KernelRuntimeData& tunedKernelData,
        const std::vector<std::pair<size_t, ThreadSizeUsage>>& additionalKernelData, const KernelConfiguration& currentConfiguration) const;
    bool validateResult(const Kernel* kernel, const KernelRunResult& result, const uint64_t manipulatorDuration,
        const KernelConfiguration& kernelConfiguration);
};

} // namespace ktt
