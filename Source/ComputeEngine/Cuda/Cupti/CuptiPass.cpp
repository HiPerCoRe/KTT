#ifdef KTT_PROFILING_CUPTI

#include <cupti_profiler_target.h>

#include <ComputeEngine/Cuda/Cupti/CuptiPass.h>
#include <ComputeEngine/Cuda/CudaUtility.h>

namespace ktt
{

CuptiPass::CuptiPass(CuptiInstance& instance) :
    m_Instance(instance)
{
    CUpti_Profiler_BeginPass_Params beginParams =
    {
        CUpti_Profiler_BeginPass_Params_STRUCT_SIZE,
        nullptr,
        instance.GetContext().GetContext()
    };

    CheckError(cuptiProfilerBeginPass(&beginParams), "cuptiProfilerBeginPass");

    CUpti_Profiler_EnableProfiling_Params enableParams =
    {
        CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE,
        nullptr,
        instance.GetContext().GetContext()
    };

    CheckError(cuptiProfilerEnableProfiling(&enableParams), "cuptiProfilerEnableProfiling");
}

CuptiPass::~CuptiPass()
{
    CUpti_Profiler_DisableProfiling_Params disableParams =
    {
        CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE,
        nullptr,
        m_Instance.GetContext().GetContext()
    };

    CheckError(cuptiProfilerDisableProfiling(&disableParams), "cuptiProfilerDisableProfiling");

    CUpti_Profiler_EndPass_Params endParams =
    {
        CUpti_Profiler_EndPass_Params_STRUCT_SIZE,
        nullptr,
        m_Instance.GetContext().GetContext()
    };

    CheckError(cuptiProfilerEndPass(&endParams), "cuptiProfilerEndPass");

    if (endParams.allPassesSubmitted)
    {
        m_Instance.CollectData();
    }
}

} // namespace ktt

#endif // KTT_PROFILING_CUPTI
