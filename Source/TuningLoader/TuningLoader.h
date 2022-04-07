#pragma once

#include <memory>
#include <string>
#include <vector>

#include <TuningLoader/TunerCommand.h>
#include <TuningLoader/TunerContext.h>

namespace ktt
{

class TuningLoader
{
public:
    void LoadTuningDescription(const std::string& file);
    void ApplyCommands();
    std::unique_ptr<Tuner> RetrieveTuner();

private:
    TunerContext m_Context;
    std::vector<std::unique_ptr<TunerCommand>> m_Commands;
};

} // namespace ktt
