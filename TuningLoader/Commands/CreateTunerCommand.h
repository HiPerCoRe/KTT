#pragma once

#include <TunerCommand.h>

namespace ktt
{

class CreateTunerCommand : public TunerCommand
{
public:
    CreateTunerCommand() = default;
    explicit CreateTunerCommand(const ComputeApi api);

    virtual void Execute(TunerContext& context) override;
    virtual CommandPriority GetPriority() const override;

private:
    ComputeApi m_Api;
};

} // namespace ktt
