#pragma once

#include <map>
#include <vector>

#include <Api/Configuration/KernelConfiguration.h>
#include <Api/Output/ComputationResult.h>
#include <Api/Output/KernelResult.h>
#include <ComputeEngine/KernelComputeData.h>
#include <Kernel/Kernel.h>
#include <KernelArgument/KernelArgument.h>
#include <KernelRunner/KernelRunMode.h>
#include <KttTypes.h>

namespace ktt
{

class ComputeLayerData
{
public:
    explicit ComputeLayerData(const Kernel& kernel, const KernelConfiguration& configuration, const KernelRunMode runMode);

    void IncreaseOverhead(const Nanoseconds overhead);
    void AddPartialResult(const ComputationResult& result);
    void AddArgumentOverride(const ArgumentId id, const KernelArgument& argument);
    void SwapArguments(const KernelDefinitionId id, const ArgumentId first, const ArgumentId second);
    void ChangeArguments(const KernelDefinitionId id, std::vector<KernelArgument*>& arguments);

    bool IsProfilingEnabled(const KernelDefinitionId id) const;
    const Kernel& GetKernel() const;
    const KernelConfiguration& GetConfiguration() const;
    KernelRunMode GetRunMode() const;
    const KernelComputeData& GetComputeData(const KernelDefinitionId id) const;
    KernelResult GenerateResult(const Nanoseconds launcherDuration) const;

private:
    std::map<KernelDefinitionId, KernelComputeData> m_ComputeData;
    std::map<ArgumentId, KernelArgument> m_ArgumentOverrides;
    std::vector<ComputationResult> m_PartialResults;
    const Kernel& m_Kernel;
    const KernelConfiguration& m_Configuration;
    KernelRunMode m_RunMode;
    Nanoseconds m_Overhead;

    Nanoseconds CalculateLauncherOverhead() const;
};

} // namespace ktt
