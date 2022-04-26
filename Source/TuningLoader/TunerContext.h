#pragma once

#include <memory>

#include <Tuner.h>

namespace ktt
{

class TunerContext
{
public:
    TunerContext();

    void SetTuner(std::unique_ptr<Tuner> tuner);
    void SetKernelDefinitionId(const KernelDefinitionId id);
    void SetKernelId(const KernelId id);
    void SetResults(const std::vector<KernelResult> results);

    Tuner& GetTuner();
    KernelDefinitionId GetKernelDefinitionId() const;
    KernelId GetKernelId() const;
    const std::vector<KernelResult>& GetResults() const;

private:
    std::unique_ptr<Tuner> m_Tuner;
    KernelDefinitionId m_DefinitionId;
    KernelId m_KernelId;
    std::vector<KernelResult> m_Results;
};

} // namespace ktt
