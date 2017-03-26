#pragma once

#include <memory>
#include <vector>

#include "../dtos/tuning_result.h"
#include "../compute_api_drivers/opencl_core.h"
#include "../kernel/kernel_manager.h"
#include "../kernel_argument/argument_manager.h"
#include "searcher.h"

namespace ktt
{

class TuningRunner
{
public:
    // Constructor
    TuningRunner(ArgumentManager* argumentManager, KernelManager* kernelManager, OpenCLCore* openCLCore);

    // Core methods
    std::vector<TuningResult> tuneKernel(const size_t id);

private:
    // Attributes
    ArgumentManager* argumentManager;
    KernelManager* kernelManager;
    OpenCLCore* openCLCore;

    // Helper methods
    std::unique_ptr<Searcher> getSearcher(const SearchMethod& searchMethod, const std::vector<double>& searchArguments,
        const std::vector<KernelConfiguration>& configurations, const std::vector<KernelParameter>& parameters) const;
    std::vector<size_t> convertDimensionVector(const DimensionVector& vector) const;
    std::vector<KernelArgument> getKernelArguments(const size_t kernelId) const;
};

} // namespace ktt
