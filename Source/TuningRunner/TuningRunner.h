#pragma once

#include <memory>
#include <string>

#include <Api/Output/BufferOutputDescriptor.h>
#include <Api/Output/KernelResult.h>
#include <Api/StopCondition/StopCondition.h>
#include <Kernel/Kernel.h>
#include <KernelRunner/KernelRunner.h>
#include <TuningRunner/ConfigurationManager.h>

namespace ktt
{

class TuningRunner
{
public:
    explicit TuningRunner(KernelRunner& kernelRunner);

    std::vector<KernelResult> Tune(const Kernel& kernel, const KernelDimensions& dimensions, std::unique_ptr<StopCondition> stopCondition);
    KernelResult TuneIteration(const Kernel& kernel, const KernelDimensions& dimensions, const KernelRunMode mode,
        const std::vector<BufferOutputDescriptor>& output, const bool recomputeReference);
    std::vector<KernelResult> SimulateTuning(const Kernel& kernel, const std::vector<KernelResult>& results, const uint64_t iterations);

    void SetSearcher(const KernelId id, std::unique_ptr<Searcher> searcher);
    void InitializeConfigurationData(const Kernel& kernel);
    void ClearConfigurationData(const KernelId id, const bool clearSearcher = false);
    uint64_t GetConfigurationsCount(const KernelId id) const;
    KernelConfiguration GetBestConfiguration(const KernelId id) const;

private:
    KernelRunner& m_KernelRunner;
    std::unique_ptr<ConfigurationManager> m_ConfigurationManager;

    static const KernelResult& FindMatchingResult(const std::vector<KernelResult>& results, const KernelConfiguration& configuration);
};

} // namespace ktt
