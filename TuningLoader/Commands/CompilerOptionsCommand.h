#pragma once

#include <string>

#include <TunerCommand.h>

namespace ktt
{

class CompilerOptionsCommand : public TunerCommand
{
public:
    CompilerOptionsCommand() = default;
    explicit CompilerOptionsCommand(const std::string& options);

    virtual void Execute(TunerContext& context) override;
    virtual CommandPriority GetPriority() const override;

private:
    std::string m_Options;
};

} // namespace ktt
