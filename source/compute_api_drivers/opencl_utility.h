#pragma once

#include <string>

#include "CL/cl.h"
#include "../enums/argument_memory_type.h"

namespace ktt
{

std::string getOpenCLEnumName(const cl_int value);
void checkOpenCLError(const cl_int value);
void checkOpenCLError(const cl_int value, const std::string& message);
cl_mem_flags getOpenCLMemoryType(const ArgumentMemoryType& argumentMemoryType);
cl_ulong getKernelRunDuration(const cl_event profilingEvent);

} // namespace ktt
