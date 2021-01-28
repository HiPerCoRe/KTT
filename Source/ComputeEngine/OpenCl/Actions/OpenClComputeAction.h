#pragma once

#ifdef KTT_API_OPENCL

#include <memory>
#include <string>

#include <Api/Output/ComputationResult.h>
#include <ComputeEngine/OpenCl/OpenClEvent.h>
#include <ComputeEngine/OpenCl/OpenClKernel.h>
#include <KttTypes.h>

namespace ktt
{

class OpenClComputeAction
{
public:
    OpenClComputeAction(const ComputeActionId id, std::shared_ptr<OpenClKernel> kernel);

    void IncreaseOverhead(const Nanoseconds overhead);
    void SetConfigurationPrefix(const std::string& prefix);
    void SetReleaseFlag();
    void WaitForFinish();

    ComputeActionId GetId() const;
    OpenClKernel& GetKernel();
    cl_event* GetEvent();
    Nanoseconds GetDuration() const;
    Nanoseconds GetOverhead() const;
    const std::string& GetConfigurationPrefix() const;
    ComputationResult GenerateResult() const;

private:
    ComputeActionId m_Id;
    std::shared_ptr<OpenClKernel> m_Kernel;
    std::unique_ptr<OpenClEvent> m_Event;
    Nanoseconds m_Overhead;
    std::string m_Prefix;
};

} // namespace ktt

#endif // KTT_API_OPENCL
