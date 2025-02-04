#pragma once

#include <string>
#include <vector>

#include <TunerCommand.h>

namespace ktt
{

class CompilerOptionsCommand : public TunerCommand
{
public:
    CompilerOptionsCommand() = default;
    explicit CompilerOptionsCommand(const std::vector<std::string>& options);

    virtual void Execute(TunerContext& context) override;
    virtual CommandPriority GetPriority() const override;

private:
    std::vector<std::string> m_Options;
};

} // namespace ktt
