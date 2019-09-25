#pragma once

#ifdef KTT_PLATFORM_OPENCL

#include <string>
#include <CL/cl.h>
#include <enum/argument_access_type.h>

#ifdef KTT_PROFILING_AMD
#include <gpu_perf_api/GPUPerfAPI.h>
#endif // KTT_PROFILING_AMD

namespace ktt
{

std::string getOpenCLEnumName(const cl_int value);
void checkOpenCLError(const cl_int value);
void checkOpenCLError(const cl_int value, const std::string& message);
cl_mem_flags getOpenCLMemoryType(const ArgumentAccessType accessType);
std::string getPlatformInfoString(const cl_platform_id id, const cl_platform_info info);
std::string getDeviceInfoString(const cl_device_id id, const cl_device_info info);

#ifdef KTT_PROFILING_AMD
void checkGPAError(const GPA_Status value, const std::string& message);
#endif // KTT_PROFILING_AMD

} // namespace ktt

#endif // KTT_PLATFORM_OPENCL
