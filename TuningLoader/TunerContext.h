#pragma once

#include <memory>

#include <Ktt.h>

namespace ktt
{

class TunerContext
{
public:
    TunerContext();

    void SetWorkingDirectory(const std::string& directory);
    void SetTuner(std::unique_ptr<Tuner> tuner);
    void SetStopCondition(std::unique_ptr<StopCondition> condition);
    void SetKernelDefinitionId(const KernelDefinitionId id);
    void SetKernelId(const KernelId id);
    void SetArguments(const std::vector<ArgumentId>& arguments);
    void SetResults(const std::vector<KernelResult> results);

    const std::string& GetWorkingDirectory() const;
    std::string GetFullPath(const std::string& relativePath) const;
    Tuner& GetTuner();
    std::unique_ptr<StopCondition> RetrieveStopCondition();
    KernelDefinitionId GetKernelDefinitionId() const;
    KernelId GetKernelId() const;
    std::vector<ArgumentId>& GetArguments();
    const std::vector<ArgumentId>& GetArguments() const;
    std::vector<ArgumentId>& GetReferenceArguments();
    const std::vector<ArgumentId>& GetReferenceArguments() const;
    const std::vector<KernelResult>& GetResults() const;

private:
    std::string m_WorkingDirectory;
    std::unique_ptr<Tuner> m_Tuner;
    std::unique_ptr<StopCondition> m_StopCondition;
    KernelDefinitionId m_DefinitionId;
    KernelId m_KernelId;
    std::vector<ArgumentId> m_Arguments;
    std::vector<ArgumentId> m_ReferenceArguments;
    std::vector<KernelResult> m_Results;
};

} // namespace ktt
