#pragma once

#include <string>
#include <vector>

#include <TunerCommand.h>

namespace ktt
{

class ProfilingCommand : public TunerCommand
{
public:
    ProfilingCommand() = default;
    explicit ProfilingCommand(const bool profilingOn);

    virtual void Execute(TunerContext& context) override;
    virtual CommandPriority GetPriority() const override;

private:
    bool m_ProfilingOn;
};

} // namespace ktt
