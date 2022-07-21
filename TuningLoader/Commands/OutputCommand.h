#pragma once

#include <string>

#include <TunerCommand.h>

namespace ktt
{

class OutputCommand : public TunerCommand
{
public:
    OutputCommand() = default;
    explicit OutputCommand(const std::string& file, const OutputFormat format);

    virtual void Execute(TunerContext& context) override;
    virtual CommandPriority GetPriority() const override;

private:
    std::string m_File;
    OutputFormat m_Format;
};

} // namespace ktt
