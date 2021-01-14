#ifdef KTT_API_OPENCL

#include <ComputeEngine/OpenCl/OpenClEvent.h>
#include <ComputeEngine/OpenCl/OpenClUtility.h>

namespace ktt
{

OpenClEvent::OpenClEvent() :
    m_Event(nullptr),
    m_ReleaseFlag(false)
{}

OpenClEvent::~OpenClEvent()
{
    if (m_ReleaseFlag)
    {
        CheckError(clReleaseEvent(m_Event), "clReleaseEvent");
    }
}

void OpenClEvent::SetReleaseFlag()
{
    m_ReleaseFlag = true;
}

void OpenClEvent::WaitForFinish()
{
    CheckError(clWaitForEvents(1, &m_Event), "clWaitForEvents");
}

cl_event* OpenClEvent::GetEvent()
{
    return &m_Event;
}

Nanoseconds OpenClEvent::GetDuration() const
{
    cl_ulong start;
    cl_ulong end;
    CheckError(clGetEventProfilingInfo(m_Event, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr),
        "clGetEventProfilingInfo");
    CheckError(clGetEventProfilingInfo(m_Event, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr),
        "clGetEventProfilingInfo");

    return static_cast<Nanoseconds>(end - start);
}

} // namespace ktt

#endif // KTT_API_OPENCL
