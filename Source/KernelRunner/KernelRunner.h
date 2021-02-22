#pragma once

#include <memory>

#include <Api/Configuration/KernelConfiguration.h>
#include <Api/Output/BufferOutputDescriptor.h>
#include <Api/Output/KernelResult.h>
#include <ComputeEngine/ComputeEngine.h>
#include <Kernel/Kernel.h>
#include <KernelArgument/KernelArgumentManager.h>
#include <KernelRunner/ComputeLayer.h>
#include <KernelRunner/KernelRunMode.h>
#include <KernelRunner/ResultValidator.h>
#include <Utility/Timer/TimeUnit.h>
#include <KttTypes.h>

namespace ktt
{

class KernelRunner
{
public:
    explicit KernelRunner(ComputeEngine& engine, KernelArgumentManager& argumentManager);

    KernelResult RunKernel(const Kernel& kernel, const KernelConfiguration& configuration, const KernelRunMode mode,
        const std::vector<BufferOutputDescriptor>& output);
    void SetupBuffers(const Kernel& kernel);
    void CleanupBuffers(const Kernel& kernel);
    void DownloadBuffers(const std::vector<BufferOutputDescriptor>& output);

    void SetTimeUnit(const TimeUnit unit);
    void SetReadOnlyArgumentCache(const bool flag);
    void SetProfiling(const bool flag);
    bool IsProfilingActive() const;

    void SetValidationMethod(const ValidationMethod method, const double toleranceThreshold);
    void SetValidationMode(const ValidationMode mode);
    void SetValidationRange(const ArgumentId id, const size_t range);
    void SetValueComparator(const ArgumentId id, ValueComparator comparator);
    void SetReferenceComputation(const ArgumentId id, ReferenceComputation computation);
    void SetReferenceKernel(const ArgumentId id, const Kernel& kernel, const KernelConfiguration& configuration);
    void ClearReferenceResult(const Kernel& kernel);

private:
    std::unique_ptr<ComputeLayer> m_ComputeLayer;
    std::unique_ptr<ResultValidator> m_Validator;
    ComputeEngine& m_Engine;
    KernelArgumentManager& m_ArgumentManager;
    TimeUnit m_TimeUnit;
    bool m_ReadOnlyCacheFlag;
    bool m_ProfilingFlag;

    KernelLauncher GetKernelLauncher(const Kernel& kernel);
    KernelResult RunKernelInternal(const Kernel& kernel, const KernelConfiguration& configuration, KernelLauncher launcher,
        const std::vector<BufferOutputDescriptor>& output);
    Nanoseconds RunLauncher(KernelLauncher launcher);

    void PrepareValidationData(const ArgumentId id);
    void ValidateResult(const Kernel& kernel, KernelResult& result, const KernelRunMode mode);
};

} // namespace ktt
