#include <ComputeEngine/EngineConfiguration.h>

namespace ktt
{

EngineConfiguration::EngineConfiguration() :
    EngineConfiguration(GlobalSizeType::OpenCL)
{}

EngineConfiguration::EngineConfiguration(const GlobalSizeType sizeType) :
    m_GlobalSizeType(sizeType),
    m_GlobalSizeCorrection(false)
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

} // namespace ktt
