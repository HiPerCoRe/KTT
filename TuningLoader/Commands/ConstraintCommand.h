#pragma once

#include <string>

#include <TunerCommand.h>

namespace ktt
{

class ConstraintCommand : public TunerCommand
{
public:
    ConstraintCommand() = default;
    explicit ConstraintCommand(const std::vector<std::string>& parameters, const std::string& expression);

    virtual void Execute(TunerContext& context) override;
    virtual CommandPriority GetPriority() const override;

private:
    std::vector<std::string> m_Parameters;
    std::string m_Expression;
};

} // namespace ktt
