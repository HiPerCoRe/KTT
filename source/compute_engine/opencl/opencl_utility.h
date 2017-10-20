#pragma once

#include <string>
#include "CL/cl.h"
#include "enum/argument_access_type.h"

namespace ktt
{

std::string getOpenclEnumName(const cl_int value);
void checkOpenclError(const cl_int value);
void checkOpenclError(const cl_int value, const std::string& message);
cl_mem_flags getOpenclMemoryType(const ArgumentAccessType& accessType);
cl_ulong getKernelRunDuration(const cl_event profilingEvent);
std::string getPlatformInfoString(const cl_platform_id id, const cl_platform_info info);
std::string getDeviceInfoString(const cl_device_id id, const cl_device_info info);

} // namespace ktt
