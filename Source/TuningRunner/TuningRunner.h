#pragma once

#include <memory>
#include <string>

#include <Api/Output/BufferOutputDescriptor.h>
#include <Api/Output/KernelResult.h>
#include <Api/StopCondition/StopCondition.h>
#include <Kernel/Kernel.h>
#include <KernelArgument/KernelArgumentManager.h>
#include <KernelRunner/KernelRunner.h>
#include <TuningRunner/ConfigurationManager.h>

namespace ktt
{

class DeviceInfo;

class TuningRunner
{
public:
    explicit TuningRunner(KernelRunner& kernelRunner, KernelArgumentManager& argumentManager, const DeviceInfo& info);

    std::vector<KernelResult> Tune(const Kernel& kernel, std::unique_ptr<StopCondition> stopCondition);
    KernelResult TuneIteration(const Kernel& kernel, const KernelRunMode mode, const std::vector<BufferOutputDescriptor>& output,
        const bool recomputeReference);
    std::vector<KernelResult> DryTune(const Kernel& kernel, const std::string& resultsFile, const uint64_t iterations);

    void SetSearcher(const KernelId id, std::unique_ptr<Searcher> searcher);
    void ClearData(const KernelId id);
    const KernelConfiguration& GetBestConfiguration(const KernelId id) const;

private:
    KernelRunner& m_KernelRunner;
    KernelArgumentManager& m_ArgumentManager;
    std::unique_ptr<ConfigurationManager> m_ConfigurationManager;
};

} // namespace ktt
