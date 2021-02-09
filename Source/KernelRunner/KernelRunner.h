#pragma once

#include <functional>
#include <map>
#include <memory>

#include <Api/Output/BufferOutputDescriptor.h>
#include <Api/Output/KernelResult.h>
#include <ComputeEngine/ComputeEngine.h>
#include <Kernel/Kernel.h>
#include <KernelArgument/ArgumentAccessType.h>
#include <KernelArgument/KernelArgumentManager.h>
#include <KernelRunner/ComputeLayer.h>
#include <KernelRunner/KernelRunMode.h>
// #include <KernelRunner/ResultValidator>
#include <Output/TimeUnit.h>
#include <KttTypes.h>

namespace ktt
{

class KernelRunner
{
public:
    explicit KernelRunner(ComputeEngine& engine, KernelArgumentManager& argumentManager);

    KernelResult RunKernel(const Kernel& kernel, const KernelConfiguration& configuration, const KernelRunMode mode,
        const std::vector<BufferOutputDescriptor>& output);
    void DownloadBuffers(const std::vector<BufferOutputDescriptor>& output);

    void SetTimeUnit(const TimeUnit unit);
    void SetProfiling(const bool flag);
    bool IsProfilingActive() const;

    //void SetValidationMethod(const ValidationMethod method, const double toleranceThreshold);
    //void SetValidationMode(const ValidationMode mode);
    //void SetValidationRange(const ArgumentId id, const size_t range);
    //void SetArgumentComparator(const ArgumentId id, const std::function<bool(const void*, const void*)>& comparator);
    //void SetReferenceKernel(const KernelId id, const KernelId referenceId, const std::vector<ParameterPair>& referenceConfiguration,
    //    const std::vector<ArgumentId>& validatedArgumentIds);
    //void SetReferenceClass(const KernelId id, std::unique_ptr<ReferenceClass> referenceClass, const std::vector<ArgumentId>& validatedArgumentIds);
    //void ClearReferenceResult(const KernelId id);

private:
    std::unique_ptr<ComputeLayer> m_ComputeLayer;
    ComputeEngine& m_Engine;
    KernelArgumentManager& m_ArgumentManager;
    // ResultValidator m_Validator;
    TimeUnit m_TimeUnit;
    bool m_ProfilingFlag;

    void SetupBuffers(const Kernel& kernel);
    void CleanupBuffers(const Kernel& kernel);
    KernelLauncher GetKernelLauncher(const Kernel& kernel);

    KernelResult RunKernelInternal(const Kernel& kernel, const KernelConfiguration& configuration, KernelLauncher launcher,
        const std::vector<BufferOutputDescriptor>& output);
    Nanoseconds RunLauncher(KernelLauncher launcher);
    void ValidateResult(const Kernel& kernel, KernelResult& result, const KernelRunMode mode);
};

} // namespace ktt
