#pragma once

#include <memory>

#include <Api/Configuration/KernelConfiguration.h>
#include <Api/Output/BufferOutputDescriptor.h>
#include <Api/Output/KernelResult.h>
#include <Api/ExceptionReason.h>
#include <ComputeEngine/ComputeEngine.h>
#include <Kernel/Kernel.h>
#include <KernelArgument/KernelArgumentManager.h>
#include <KernelRunner/ComputeLayer.h>
#include <KernelRunner/KernelRunMode.h>
#include <KernelRunner/ResultValidator.h>
#include <KttTypes.h>

namespace ktt
{

class KernelRunner
{
public:
    explicit KernelRunner(ComputeEngine& engine, KernelArgumentManager& argumentManager);

    KernelResult RunKernel(const Kernel& kernel, const KernelConfiguration& configuration, const KernelDimensions& dimensions,
        const KernelRunMode mode, const std::vector<BufferOutputDescriptor>& output, const bool manageBuffers = true);
    void SetupBuffers(const Kernel& kernel);
    void CleanupBuffers(const Kernel& kernel);
    void DownloadBuffers(const std::vector<BufferOutputDescriptor>& output);

    void SetReadOnlyArgumentCache(const bool flag);
    void SetProfiling(const bool flag);
    bool IsProfilingActive() const;

    void SetValidationMethod(const ValidationMethod method, const double toleranceThreshold);
    void SetValidationMode(const ValidationMode mode);
    void SetValidationRange(const ArgumentId& id, const size_t range);
    void SetValueComparator(const ArgumentId& id, ValueComparator comparator);
    void SetReferenceComputation(const ArgumentId& id, ReferenceComputation computation);
    void SetReferenceKernel(const ArgumentId& id, const Kernel& kernel, const KernelConfiguration& configuration,
        const KernelDimensions& dimensions);
    void SetReferenceArgument(const ArgumentId& id, const KernelArgument& argument);
    void ClearReferenceResult(const Kernel& kernel);
    void RemoveKernelData(const KernelId id);
    void RemoveValidationData(const ArgumentId& id);

private:
    std::unique_ptr<ComputeLayer> m_ComputeLayer;
    std::unique_ptr<ResultValidator> m_Validator;
    ComputeEngine& m_Engine;
    KernelArgumentManager& m_ArgumentManager;
    bool m_ReadOnlyCacheFlag;
    //bool m_ProfilingFlag;

    KernelLauncher GetKernelLauncher(const Kernel& kernel);
    KernelResult RunKernelInternal(const Kernel& kernel, const KernelConfiguration& configuration, const KernelDimensions& dimensions,
        const KernelRunMode mode, KernelLauncher launcher, const std::vector<BufferOutputDescriptor>& output);
    Nanoseconds RunLauncher(KernelLauncher launcher);

    void PrepareValidationData(const ArgumentId& id);
    void ValidateResult(const Kernel& kernel, KernelResult& result, const KernelRunMode mode);
    static ResultStatus GetStatusFromException(const ExceptionReason reason);
};

} // namespace ktt
