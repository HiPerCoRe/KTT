#pragma once

#include <map>
#include <memory>
#include <utility>
#include <vector>
#include <api/stop_condition/stop_condition.h>
#include <api/device_info.h>
#include <dto/kernel_result.h>
#include <kernel/kernel_manager.h>
#include <kernel_argument/argument_manager.h>
#include <tuning_runner/configuration_manager.h>
#include <tuning_runner/kernel_runner.h>
#include <utility/result_printer.h>

namespace ktt
{

class TuningRunner
{
public:
    // Constructor
    explicit TuningRunner(ArgumentManager* argumentManager, KernelManager* kernelManager, KernelRunner* kernelRunner, const DeviceInfo& info);

    // Core methods
    std::vector<ComputationResult> tuneKernel(const KernelId id, std::unique_ptr<StopCondition> stopCondition);
    std::vector<ComputationResult> dryTuneKernel(const KernelId id, const std::string& filePath, const size_t iterations);
    std::vector<ComputationResult> tuneComposition(const KernelId id, std::unique_ptr<StopCondition> stopCondition);
    ComputationResult tuneKernelByStep(const KernelId id, const KernelRunMode mode, const std::vector<OutputDescriptor>& output,
        const bool recomputeReference);
    ComputationResult tuneCompositionByStep(const KernelId id, const KernelRunMode mode, const std::vector<OutputDescriptor>& output,
        const bool recomputeReference);
    void clearKernelData(const KernelId id, const bool clearConfigurations);
    void setKernelProfiling(const bool flag);
    void setSearchMethod(const SearchMethod method, const std::vector<double>& arguments);
    ComputationResult getBestComputationResult(const KernelId id) const;

    // Result printer methods
    void setTimeUnit(const TimeUnit unit);
    void setInvalidResultPrinting(const bool flag);
    void printResult(const KernelId id, std::ostream& outputTarget, const PrintFormat format) const;
    void printResult(const KernelId id, const std::string& filePath, const PrintFormat format) const;

private:
    // Attributes
    ArgumentManager* argumentManager;
    KernelManager* kernelManager;
    KernelRunner* kernelRunner;
    ConfigurationManager configurationManager;
    ResultPrinter resultPrinter;

    // Helper methods
    bool hasWritableZeroCopyArguments(const Kernel& kernel) const;
};

} // namespace ktt
