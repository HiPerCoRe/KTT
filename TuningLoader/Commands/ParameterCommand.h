#pragma once

#include <string>

#include <TunerCommand.h>

namespace ktt
{

class ParameterCommand : public TunerCommand
{
public:
    ParameterCommand() = default;
    explicit ParameterCommand(const std::string& name, const std::string& valueType, const std::string& valueScript);

    virtual void Execute(TunerContext& context) override;
    virtual CommandPriority GetPriority() const override;

private:
    std::string m_Name;
    std::string m_ValueScript;
    ParameterValueType m_ValueType;

    static ParameterValueType GetValueTypeFromString(const std::string& valueType);
};

} // namespace ktt
