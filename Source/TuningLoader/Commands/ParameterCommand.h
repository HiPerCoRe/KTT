#pragma once

#include <string>

#include <TuningLoader/TunerCommand.h>

namespace ktt
{

class ParameterCommand : public TunerCommand
{
public:
    ParameterCommand() = default;
    explicit ParameterCommand(const std::string& name, const std::vector<ParameterValue>& values);

    virtual void Execute(TunerContext& context) override;
    virtual CommandPriority GetPriority() const override;

private:
    std::string m_Name;
    std::vector<ParameterValue> m_Values;
};

} // namespace ktt
