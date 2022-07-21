#pragma once

#include <string>

#include <TunerCommand.h>

namespace ktt
{

class ModifierCommand : public TunerCommand
{
public:
    ModifierCommand() = default;
    explicit ModifierCommand(const ModifierType type, const ModifierDimension dimension, const std::string& parameter,
        const ModifierAction action);

    virtual void Execute(TunerContext& context) override;
    virtual CommandPriority GetPriority() const override;

private:
    ModifierType m_Type;
    ModifierDimension m_Dimension;
    std::string m_Parameter;
    ModifierAction m_Action;
};

} // namespace ktt
