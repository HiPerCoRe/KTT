#pragma once

#include <string>

#include <TuningLoader/TunerCommand.h>

namespace ktt
{

class AddKernelCommand : public TunerCommand
{
public:
    AddKernelCommand() = default;
    explicit AddKernelCommand(const std::string& name, const std::string& file, const DimensionVector& globalSize);

    virtual void Execute(TunerContext& context) override;
    virtual CommandPriority GetPriority() const override;

private:
    std::string m_Name;
    std::string m_File;
    DimensionVector m_GlobalSize;
};

} // namespace ktt
