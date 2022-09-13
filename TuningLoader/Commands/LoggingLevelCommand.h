#pragma once

#include <TunerCommand.h>

namespace ktt
{

class LoggingLevelCommand : public TunerCommand
{
public:
    LoggingLevelCommand() = default;
    explicit LoggingLevelCommand(const LoggingLevel level);

    virtual void Execute(TunerContext& context) override;
    virtual CommandPriority GetPriority() const override;

private:
    LoggingLevel m_LoggingLevel;
};

} // namespace ktt
