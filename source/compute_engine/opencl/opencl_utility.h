#pragma once

#ifdef KTT_PLATFORM_OPENCL

#include <string>
#include <CL/cl.h>
#include <enum/argument_access_type.h>

#ifdef KTT_PROFILING_GPA
#include <gpu_perf_api/GPUPerfAPI.h>
#endif // KTT_PROFILING_GPA

#ifdef KTT_PROFILING_GPA_LEGACY
#include <gpu_perf_api_legacy/GPUPerfAPI.h>
#endif // KTT_PROFILING_GPA_LEGACY

namespace ktt
{

std::string getOpenCLEnumName(const cl_int value);
void checkOpenCLError(const cl_int value, const std::string& message);
cl_mem_flags getOpenCLMemoryType(const ArgumentAccessType accessType);
std::string getPlatformInfoString(const cl_platform_id id, const cl_platform_info info);
std::string getDeviceInfoString(const cl_device_id id, const cl_device_info info);

#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
void checkGPAError(const GPA_Status value, const std::string& message, GPAFunctionTable& gpaFunctions);
#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY

} // namespace ktt

#endif // KTT_PLATFORM_OPENCL
