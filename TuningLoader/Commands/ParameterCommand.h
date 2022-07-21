#pragma once

#include <string>

#include <TunerCommand.h>

namespace ktt
{

class ParameterCommand : public TunerCommand
{
public:
    ParameterCommand() = default;
    explicit ParameterCommand(const ParameterValueType valueType, const std::string& name, const std::vector<ParameterValue>& values);

    virtual void Execute(TunerContext& context) override;
    virtual CommandPriority GetPriority() const override;

private:
    ParameterValueType m_ValueType;
    std::string m_Name;
    std::vector<ParameterValue> m_Values;
};

} // namespace ktt
