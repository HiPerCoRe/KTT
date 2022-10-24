#pragma once

#include <TunerCommand.h>

namespace ktt
{

class ValidationCommand : public TunerCommand
{
public:
    ValidationCommand() = default;
    explicit ValidationCommand(const ArgumentId& id, const ArgumentId& referenceId);

    virtual void Execute(TunerContext& context) override;
    virtual CommandPriority GetPriority() const override;

private:
    ArgumentId m_Id;
    ArgumentId m_ReferencedId;
};

} // namespace ktt
