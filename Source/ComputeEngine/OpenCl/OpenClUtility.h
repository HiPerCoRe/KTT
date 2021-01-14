#pragma once

#ifdef KTT_API_OPENCL

#include <string>
#include <CL/cl.h>

#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
#include <GPUPerfAPI.h>
#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY

#include <KernelArgument/ArgumentAccessType.h>

namespace ktt
{

std::string GetEnumName(const cl_int value);
void CheckError(const cl_int value, const std::string& function, const std::string& info = "");

#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
void CheckError(const GPA_Status value, GPAFunctionTable& gpaFunctions, const std::string& function, const std::string& info = "");
#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY

} // namespace ktt

#endif // KTT_API_OPENCL
