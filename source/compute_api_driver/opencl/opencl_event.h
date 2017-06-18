#pragma once

#include "CL/cl.h"
#include "opencl_utility.h"

namespace ktt
{

class OpenclEvent
{
public:
    OpenclEvent(const cl_context context) :
        context(context)
    {
        cl_int result;
        event = clCreateUserEvent(context, &result);
        checkOpenclError(result, std::string("clCreateUserEvent"));
    }

    ~OpenclEvent()
    {
        checkOpenclError(clReleaseEvent(event), std::string("clReleaseEvent"));
    }

    cl_context getContext() const
    {
        return context;
    }

    cl_event getEvent() const
    {
        return event;
    }

private:
    cl_context context;
    cl_event event;
};

} // namespace ktt
