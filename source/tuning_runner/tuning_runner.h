#pragma once

#include <memory>
#include <vector>

#include "../dto/tuning_result.h"
#include "../compute_api_driver/opencl/opencl_core.h"
#include "../kernel/kernel_manager.h"
#include "../kernel_argument/argument_manager.h"
#include "result_validator.h"
#include "searcher/searcher.h"

namespace ktt
{

class TuningRunner
{
public:
    // Constructor
    TuningRunner(ArgumentManager* argumentManager, KernelManager* kernelManager, OpenCLCore* openCLCore);

    // Core methods
    std::vector<TuningResult> tuneKernel(const size_t id);
    void setValidationMethod(const ValidationMethod& validationMethod, const double toleranceThreshold);

private:
    // Attributes
    ArgumentManager* argumentManager;
    KernelManager* kernelManager;
    OpenCLCore* openCLCore;
    ResultValidator resultValidator;

    // Helper methods
    std::unique_ptr<Searcher> getSearcher(const SearchMethod& searchMethod, const std::vector<double>& searchArguments,
        const std::vector<KernelConfiguration>& configurations, const std::vector<KernelParameter>& parameters) const;
    std::vector<size_t> convertDimensionVector(const DimensionVector& vector) const;
    std::vector<KernelArgument> getKernelArguments(const size_t kernelId) const;
    bool validateResult(const Kernel* kernel, const KernelRunResult& result);
    bool validateResult(const Kernel* kernel, const KernelRunResult& result, bool useReferenceClass);
    bool argumentIndexExists(const size_t argumentIndex, const std::vector<size_t>& argumentIndices) const;
    std::vector<KernelArgument> getReferenceResultFromClass(const ReferenceClass* referenceClass,
        const std::vector<size_t>& referenceArgumentIndices) const;
    std::vector<KernelArgument> getReferenceResultFromKernel(const size_t referenceKernelId,
        const std::vector<ParameterValue>& referenceKernelConfiguration, const std::vector<size_t>& referenceArgumentIndices) const;
};

} // namespace ktt
