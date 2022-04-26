#pragma once

#include <string>

#include <TuningLoader/TunerCommand.h>

namespace ktt
{

class TimeUnitCommand : public TunerCommand
{
public:
    TimeUnitCommand() = default;
    explicit TimeUnitCommand(const TimeUnit timeUnit);

    virtual void Execute(TunerContext& context) override;
    virtual CommandPriority GetPriority() const override;

private:
    TimeUnit m_TimeUnit;
};

} // namespace ktt
