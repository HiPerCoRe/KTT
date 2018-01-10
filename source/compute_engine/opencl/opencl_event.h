#pragma once

#include "CL/cl.h"
#include "opencl_utility.h"

namespace ktt
{

class OpenclEvent
{
public:
	~OpenclEvent()
	{
		checkOpenclError(clReleaseEvent(event), "clReleaseEvent");
	}

	cl_event* getEvent()
	{
		return &event;
	}

	cl_ulong getKernelRunDuration()
	{
		cl_ulong start;
		cl_ulong end;
		checkOpenclError(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, nullptr), "clGetEventProfilingInfo");
		checkOpenclError(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, nullptr), "clGetEventProfilingInfo");

		return end - start;
	}

private:
	cl_event event;
};

} // namespace ktt
