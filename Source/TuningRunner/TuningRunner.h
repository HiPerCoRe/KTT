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

    std::vector<KernelResult> Tune(const Kernel& kernel, std::unique_ptr<StopCondition> stopCondition);
    KernelResult TuneIteration(const Kernel& kernel, const KernelRunMode mode, const std::vector<BufferOutputDescriptor>& output,
        const bool recomputeReference);
    void SimulateTuning(const Kernel& kernel, const std::vector<KernelResult>& results, const uint64_t iterations);

    void SetSearcher(const KernelId id, std::unique_ptr<Searcher> searcher);
    void ClearData(const KernelId id);
    KernelConfiguration GetBestConfiguration(const KernelId id) const;

private:
    KernelRunner& m_KernelRunner;
    std::unique_ptr<ConfigurationManager> m_ConfigurationManager;

    static const KernelResult& FindMatchingResult(const std::vector<KernelResult>& results, const KernelConfiguration& configuration);
};

} // namespace ktt
