#pragma once

#ifdef KTT_PLATFORM_OPENCL

#include <string>
#include <CL/cl.h>
#include <enum/argument_access_type.h>

namespace ktt
{

std::string getOpenCLEnumName(const cl_int value);
void checkOpenCLError(const cl_int value);
void checkOpenCLError(const cl_int value, const std::string& message);
cl_mem_flags getOpenCLMemoryType(const ArgumentAccessType accessType);
std::string getPlatformInfoString(const cl_platform_id id, const cl_platform_info info);
std::string getDeviceInfoString(const cl_device_id id, const cl_device_info info);

} // namespace ktt

#endif // KTT_PLATFORM_OPENCL
