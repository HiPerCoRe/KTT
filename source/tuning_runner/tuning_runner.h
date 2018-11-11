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
#include <tuning_runner/result_validator.h>

namespace ktt
{

class TuningRunner
{
public:
    // Constructor
    explicit TuningRunner(ArgumentManager* argumentManager, KernelManager* kernelManager, KernelRunner* kernelRunner, const DeviceInfo& info);

    // Core methods
    std::vector<KernelResult> tuneKernel(const KernelId id, std::unique_ptr<StopCondition> stopCondition);
    std::vector<KernelResult> dryTuneKernel(const KernelId id, const std::string& filePath, const size_t iterations);
    std::vector<KernelResult> tuneComposition(const KernelId id, std::unique_ptr<StopCondition> stopCondition);
    KernelResult tuneKernelByStep(const KernelId id, const std::vector<OutputDescriptor>& output, const bool recomputeReference);
    KernelResult tuneCompositionByStep(const KernelId id, const std::vector<OutputDescriptor>& output, const bool recomputeReference);
    void clearKernelData(const KernelId id, const bool clearConfigurations);
    void setSearchMethod(const SearchMethod method, const std::vector<double>& arguments);
    void setValidationMethod(const ValidationMethod method, const double toleranceThreshold);
    void setValidationRange(const ArgumentId id, const size_t range);
    void setArgumentComparator(const ArgumentId id, const std::function<bool(const void*, const void*)>& comparator);
    void setReferenceKernel(const KernelId id, const KernelId referenceId, const std::vector<ParameterPair>& referenceConfiguration,
        const std::vector<ArgumentId>& validatedArgumentIds);
    void setReferenceClass(const KernelId id, std::unique_ptr<ReferenceClass> referenceClass, const std::vector<ArgumentId>& validatedArgumentIds);
    ComputationResult getBestComputationResult(const KernelId id) const;

private:
    // Attributes
    ArgumentManager* argumentManager;
    KernelManager* kernelManager;
    KernelRunner* kernelRunner;
    ConfigurationManager configurationManager;
    ResultValidator resultValidator;

    // Helper methods
    bool validateResult(const Kernel& kernel, const KernelResult& result);
    bool hasWritableZeroCopyArguments(const Kernel& kernel);
};

} // namespace ktt
