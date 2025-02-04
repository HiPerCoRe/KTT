#pragma once

#include <vector>

#include <Commands/AddArgumentCommand.h>
#include <TunerCommand.h>

namespace ktt
{

class ValidationCommand : public TunerCommand
{
public:
    ValidationCommand() = default;
    explicit ValidationCommand(const ArgumentId& referenceId, const AddArgumentCommand& command, const ValidationMethod validationMethod, const double validationThreshold);

    virtual void Execute(TunerContext& context) override;
    virtual CommandPriority GetPriority() const override;

    void SetReferenceProperties(const std::vector<AddArgumentCommand>& commands);

private:
    ArgumentId m_ReferencedId;
    AddArgumentCommand m_ArgumentCommand;
    ValidationMethod m_ValidationMethod;
    double m_ValidationThreshold;
};

} // namespace ktt
