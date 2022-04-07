#pragma once

#include <TuningLoader/CommandPriority.h>
#include <TuningLoader/TunerContext.h>
#include <Tuner.h>

namespace ktt
{

class TunerCommand
{
public:
    virtual ~TunerCommand() = default;

    virtual void Execute(TunerContext& context) = 0;
    virtual CommandPriority GetPriority() const = 0;
};

} // namespace ktt
