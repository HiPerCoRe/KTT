#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "manipulator_interface_implementation.h"
#include "result_validator.h"
#include "searcher/searcher.h"
#include "../compute_api_driver/compute_api_driver.h"
#include "../dto/tuning_result.h"
#include "../kernel/kernel_manager.h"
#include "../kernel_argument/argument_manager.h"
#include "../utility/logger.h"

namespace ktt
{

class TuningRunner
{
public:
    // Constructor
    explicit TuningRunner(ArgumentManager* argumentManager, KernelManager* kernelManager, Logger* logger, ComputeApiDriver* computeApiDriver);

    // Core methods
    std::vector<TuningResult> tuneKernel(const size_t id);
    void setValidationMethod(const ValidationMethod& validationMethod, const double toleranceThreshold);

private:
    // Attributes
    ArgumentManager* argumentManager;
    KernelManager* kernelManager;
    Logger* logger;
    ComputeApiDriver* computeApiDriver;
    ResultValidator resultValidator;
    std::unique_ptr<ManipulatorInterfaceImplementation> manipulatorInterfaceImplementation;

    // Helper methods
    std::pair<KernelRunResult, uint64_t> runKernel(Kernel* kernel, const KernelConfiguration& currentConfiguration,
        const size_t currentConfigurationIndex, const size_t configurationsCount);
    std::pair<KernelRunResult, uint64_t> runKernelWithManipulator(TuningManipulator* manipulator,
        const std::vector<std::pair<size_t, KernelRuntimeData>>& kernelDataVector, const KernelConfiguration& currentConfiguration);
    std::unique_ptr<Searcher> getSearcher(const SearchMethod& searchMethod, const std::vector<double>& searchArguments,
        const std::vector<KernelConfiguration>& configurations, const std::vector<KernelParameter>& parameters) const;
    std::vector<KernelArgument> getKernelArguments(const size_t kernelId) const;
    std::vector<std::pair<size_t, KernelRuntimeData>> getKernelDataVector(const size_t tunedKernelId, const KernelRuntimeData& tunedKernelData,
        const std::vector<size_t>& additionalKernelIds, const KernelConfiguration& currentConfiguration) const;
    bool processResult(const Kernel* kernel, const KernelRunResult& result, const uint64_t manipulatorDuration);
    bool validateResult(const Kernel* kernel, const KernelRunResult& result);
    bool validateResult(const Kernel* kernel, const KernelRunResult& result, bool useReferenceClass);
    bool argumentExists(const KernelArgument& argument, const std::vector<KernelArgument>& arguments) const;
    bool argumentIndexExists(const size_t argumentIndex, const std::vector<size_t>& argumentIndices) const;
    std::vector<KernelArgument> getReferenceResultFromClass(const ReferenceClass* referenceClass,
        const std::vector<size_t>& referenceArgumentIndices) const;
    std::vector<KernelArgument> getReferenceResultFromKernel(const size_t referenceKernelId,
        const std::vector<ParameterValue>& referenceKernelConfiguration, const std::vector<size_t>& referenceArgumentIndices) const;
    void printArgument(const KernelArgument& kernelArgument, const std::string& kernelName) const;
};

} // namespace ktt
