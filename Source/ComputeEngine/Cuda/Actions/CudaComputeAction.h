#pragma once

#ifdef KTT_API_CUDA

#include <memory>
#include <string>

#include <Api/Output/KernelResult.h>
#include <ComputeEngine/Cuda/CudaEvent.h>
#include <ComputeEngine/Cuda/CudaKernel.h>
#include <KttTypes.h>

namespace ktt
{

class CudaComputeAction
{
public:
    CudaComputeAction(const ComputeActionId id, std::shared_ptr<CudaKernel> kernel);

    void IncreaseOverhead(const Nanoseconds overhead);
    void SetConfigurationPrefix(const std::string& prefix);
    void WaitForFinish();

    ComputeActionId GetId() const;
    CudaKernel& GetKernel();
    CUevent GetStartEvent() const;
    CUevent GetEndEvent() const;
    Nanoseconds GetDuration() const;
    Nanoseconds GetOverhead() const;
    const std::string& GetConfigurationPrefix() const;
    KernelResult GenerateResult() const;

private:
    ComputeActionId m_Id;
    std::shared_ptr<CudaKernel> m_Kernel;
    std::unique_ptr<CudaEvent> m_StartEvent;
    std::unique_ptr<CudaEvent> m_EndEvent;
    Nanoseconds m_Overhead;
    std::string m_Prefix;
};

} // namespace ktt

#endif // KTT_API_CUDA
