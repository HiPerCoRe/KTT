#pragma once

#include <string>
#include <vector>

#include <TunerCommand.h>

namespace ktt
{

class AddKernelCommand : public TunerCommand
{
public:
    AddKernelCommand() = default;
    explicit AddKernelCommand(const std::string& name, const std::string& file, const std::vector<std::string>& typeNames = {});

    virtual void Execute(TunerContext& context) override;
    virtual CommandPriority GetPriority() const override;

private:
    std::string m_Name;
    std::string m_File;
    std::vector<std::string> m_TypeNames;
};

} // namespace ktt
