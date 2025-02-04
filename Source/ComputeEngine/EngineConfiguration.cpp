#include <ComputeEngine/EngineConfiguration.h>

namespace ktt
{

EngineConfiguration::EngineConfiguration() :
    EngineConfiguration(GlobalSizeType::OpenCL)
{}

EngineConfiguration::EngineConfiguration(const GlobalSizeType sizeType) :
    m_GlobalSizeType(sizeType),
    m_GlobalSizeCorrection(false),
    m_ProfilingFlag(false)
{}

void EngineConfiguration::SetCompilerOptions(const std::string& options)
{
    m_CompilerOptions = options;
}

void EngineConfiguration::SetGlobalSizeType(const GlobalSizeType sizeType)
{
    m_GlobalSizeType = sizeType;
}

void EngineConfiguration::SetGlobalSizeCorrection(const bool sizeCorrection)
{
    m_GlobalSizeCorrection = sizeCorrection;
}

const std::string& EngineConfiguration::GetCompilerOptions() const
{
    return m_CompilerOptions;
}

GlobalSizeType EngineConfiguration::GetGlobalSizeType() const
{
    return m_GlobalSizeType;
}

bool EngineConfiguration::GetGlobalSizeCorrection() const
{
    return m_GlobalSizeCorrection;
}

void EngineConfiguration::SetProfiling(const bool profiling)
{
    m_ProfilingFlag = profiling;
}

bool EngineConfiguration::IsProfilingActive() const
{
    return m_ProfilingFlag;
}

} // namespace ktt
