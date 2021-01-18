#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)

#include <string>

#include <ComputeEngine/OpenCl/Gpa/GpaPass.h>
#include <ComputeEngine/OpenCl/OpenClUtility.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

GpaPass::GpaPass(GPAFunctionTable& functions, GpaInstance& instance) :
    m_Functions(functions),
    m_Instance(instance),
    m_PassIndex(instance.GetPassIndex())
{
    Logger::LogDebug("Beginning GPA pass with index " + std::to_string(m_PassIndex)
        + " for GPA instance with sample id " + std::to_string(m_Instance.GetSampleId()));
    CheckError(functions.GPA_BeginCommandList(instance.GetSession(), m_PassIndex, GPA_NULL_COMMAND_LIST, GPA_COMMAND_LIST_NONE,
        &m_CommandList), functions, "GPA_BeginCommandList");
    CheckError(functions.GPA_BeginSample(instance.GetSampleId(), m_CommandList), functions, "GPA_BeginSample");
}

GpaPass::~GpaPass()
{
    Logger::LogDebug("Ending GPA pass with index " + std::to_string(m_PassIndex)
        + " for GPA instance with sample id " + std::to_string(m_Instance.GetSampleId()));
    CheckError(m_Functions.GPA_EndSample(m_CommandList), m_Functions, "GPA_EndSample");
    CheckError(m_Functions.GPA_EndCommandList(m_CommandList), m_Functions, "GPA_EndCommandList");
    m_Instance.UpdatePassIndex();
}

} // namespace ktt

#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY
