#pragma once

#include <TunerCommand.h>

namespace ktt
{

class TuneCommand : public TunerCommand
{
public:
    TuneCommand() = default;
    TuneCommand(const std::string& simulationInput);

    virtual void Execute(TunerContext& context) override;
    virtual CommandPriority GetPriority() const override;
    OutputFormat DetectOutputFormat(std::string filePath);

private:
    std::string m_SimulationInput;
};

} // namespace ktt
