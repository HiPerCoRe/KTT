#pragma once

#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)

#include <GPUPerfAPI.h>

#include <ComputeEngine/OpenCl/Gpa/GpaInstance.h>

namespace ktt
{

class GpaPass
{
public:
    explicit GpaPass(GPAFunctionTable& functions, GpaInstance& instance);
    ~GpaPass();

private:
    GPAFunctionTable& m_Functions;
    GpaInstance& m_Instance;
    gpa_uint32 m_PassIndex;
    GPA_CommandListId m_CommandList;
};

} // namespace ktt

#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY
