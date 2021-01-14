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
    OpenClComputeAction(const ComputeActionId id);
    OpenClComputeAction(const ComputeActionId id, std::shared_ptr<OpenClKernel> kernel);

    void SetOverhead(const Nanoseconds overhead);

    ComputeActionId GetId() const;
    OpenClKernel& GetKernel();
    OpenClEvent& GetEvent();
    Nanoseconds GetOverhead() const;
    bool IsValid() const;

private:
    ComputeActionId m_Id;
    std::shared_ptr<OpenClKernel> m_Kernel;
    std::unique_ptr<OpenClEvent> m_Event;
    Nanoseconds m_Overhead;
};

} // namespace ktt

#endif // KTT_API_OPENCL
