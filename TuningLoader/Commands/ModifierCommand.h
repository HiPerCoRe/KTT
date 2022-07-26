#pragma once

#include <map>
#include <string>

#include <TunerCommand.h>

namespace ktt
{

class ModifierCommand : public TunerCommand
{
public:
    ModifierCommand() = default;
    explicit ModifierCommand(const ModifierType type, const std::map<ModifierDimension, std::string>& scripts);

    virtual void Execute(TunerContext& context) override;
    virtual CommandPriority GetPriority() const override;

    void SetType(const ModifierType type);

private:
    ModifierType m_Type;
    std::map<ModifierDimension, std::string> m_Scripts;
};

} // namespace ktt
