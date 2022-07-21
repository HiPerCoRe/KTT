#pragma once

#include <TunerCommand.h>

namespace ktt
{

class TuneCommand : public TunerCommand
{
public:
    TuneCommand() = default;

    virtual void Execute(TunerContext& context) override;
    virtual CommandPriority GetPriority() const override;
};

} // namespace ktt
