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

    Tuner& GetTuner();
    KernelDefinitionId GetKernelDefinitionId() const;
    KernelId GetKernelId() const;

    std::unique_ptr<Tuner> RetrieveTuner();

private:
    std::unique_ptr<Tuner> m_Tuner;
    KernelDefinitionId m_DefinitionId;
    KernelId m_KernelId;
};

} // namespace ktt
