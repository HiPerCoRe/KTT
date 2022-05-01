#pragma once

#include <string>

#include <TuningLoader/TunerCommand.h>

namespace ktt
{

class AddArgumentCommand : public TunerCommand
{
public:
    AddArgumentCommand() = default;
    explicit AddArgumentCommand(const std::string& name);

    virtual void Execute(TunerContext& context) override;
    virtual CommandPriority GetPriority() const override;

private:
    std::string m_Name;
};

} // namespace ktt
