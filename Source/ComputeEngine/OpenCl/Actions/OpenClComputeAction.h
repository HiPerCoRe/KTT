#pragma once

#ifdef KTT_API_OPENCL

#include <memory>

#include <ComputeEngine/OpenCl/OpenClEvent.h>
#include <ComputeEngine/OpenCl/OpenClKernel.h>
#include <KttTypes.h>

namespace ktt
{

class OpenClComputeAction
{
public:
    OpenClComputeAction(const ComputeActionId id, std::shared_ptr<OpenClKernel> kernel);

    void SetOverhead(const Nanoseconds overhead);
    void SetReleaseFlag();
    void WaitForFinish();

    ComputeActionId GetId() const;
    OpenClKernel& GetKernel();
    cl_event* GetEvent();
    Nanoseconds GetDuration() const;
    Nanoseconds GetOverhead() const;

private:
    ComputeActionId m_Id;
    std::shared_ptr<OpenClKernel> m_Kernel;
    std::unique_ptr<OpenClEvent> m_Event;
    Nanoseconds m_Overhead;
};

} // namespace ktt

#endif // KTT_API_OPENCL
