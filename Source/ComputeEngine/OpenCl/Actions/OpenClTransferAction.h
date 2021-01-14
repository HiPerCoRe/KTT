#pragma once

#ifdef KTT_API_OPENCL

#include <memory>

#include <ComputeEngine/OpenCl/OpenClEvent.h>
#include <KttTypes.h>

namespace ktt
{

class OpenClTransferAction
{
public:
    OpenClTransferAction(const TransferActionId id);

    void SetOverhead(const Nanoseconds overhead);

    TransferActionId GetId() const;
    OpenClEvent& GetEvent();
    Nanoseconds GetOverhead() const;
    bool IsValid() const;

private:
    TransferActionId m_Id;
    std::unique_ptr<OpenClEvent> m_Event;
    Nanoseconds m_Overhead;
};

} // namespace ktt

#endif // KTT_API_OPENCL
