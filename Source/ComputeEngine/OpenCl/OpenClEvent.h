#pragma once

#ifdef KTT_API_OPENCL

#include <CL/cl.h>

#include <KttTypes.h>

namespace ktt
{

class OpenClEvent
{
public:
    OpenClEvent();
    ~OpenClEvent();

    void SetReleaseFlag();
    void WaitForFinish();

    cl_event* GetEvent();
    Nanoseconds GetDuration() const;

private:
    cl_event m_Event;
    bool m_ReleaseFlag;
};

} // namespace ktt

#endif // KTT_API_OPENCL
