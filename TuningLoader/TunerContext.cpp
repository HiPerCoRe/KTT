#include <filesystem>

#include <TunerContext.h>

namespace ktt
{

TunerContext::TunerContext() :
    m_Tuner(nullptr),
    m_StopCondition(nullptr),
    m_DefinitionId(InvalidKernelDefinitionId),
    m_KernelId(InvalidKernelDefinitionId)
{}

void TunerContext::SetWorkingDirectory(const std::string& directory)
{
    m_WorkingDirectory = directory;
}

void TunerContext::SetTuner(std::unique_ptr<Tuner> tuner)
{
    m_Tuner = std::move(tuner);
}

void TunerContext::SetStopCondition(std::unique_ptr<StopCondition> condition)
{
    m_StopCondition = std::move(condition);
}

void TunerContext::SetKernelDefinitionId(const KernelDefinitionId id)
{
    m_DefinitionId = id;
}

void TunerContext::SetKernelId(const KernelId id)
{
    m_KernelId = id;
}

void TunerContext::SetArguments(const std::vector<ArgumentId>& arguments)
{
    m_Arguments = arguments;
}

void TunerContext::SetResults(const std::vector<KernelResult> results)
{
    m_Results = results;
}

const std::string& TunerContext::GetWorkingDirectory() const
{
    return m_WorkingDirectory;
}

std::string TunerContext::GetFullPath(const std::string& relativePath) const
{
    std::filesystem::path path(GetWorkingDirectory());
    path.append(relativePath);
    return path.string();
}

Tuner& TunerContext::GetTuner()
{
    return *m_Tuner;
}

std::unique_ptr<StopCondition> TunerContext::RetrieveStopCondition()
{
    return std::move(m_StopCondition);
}

KernelDefinitionId TunerContext::GetKernelDefinitionId() const
{
    return m_DefinitionId;
}

KernelId TunerContext::GetKernelId() const
{
    return m_KernelId;
}

std::vector<ArgumentId>& TunerContext::GetArguments()
{
    return m_Arguments;
}

const std::vector<ArgumentId>& TunerContext::GetArguments() const
{
    return m_Arguments;
}

const std::vector<KernelResult>& TunerContext::GetResults() const
{
    return m_Results;
}

} // namespace ktt
