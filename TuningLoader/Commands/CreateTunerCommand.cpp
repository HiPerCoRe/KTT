#include <Commands/CreateTunerCommand.h>

namespace ktt
{

CreateTunerCommand::CreateTunerCommand(const ComputeApi api) :
    m_Api(api)
{}

void CreateTunerCommand::Execute(TunerContext& context)
{
    auto tuner = std::make_unique<Tuner>(0, 0, m_Api);
    context.SetTuner(std::move(tuner));
}

CommandPriority CreateTunerCommand::GetPriority() const
{
    return CommandPriority::TunerCreation;
}

} // namespace ktt
