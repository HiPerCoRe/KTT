#include <Commands/CreateTunerCommand.h>

namespace ktt
{

CreateTunerCommand::CreateTunerCommand(const PlatformIndex platformId, const DeviceIndex deviceId, const ComputeApi api) :
    m_PlatformId(platformId),
    m_DeviceId(deviceId),
    m_Api(api)
{}

void CreateTunerCommand::Execute(TunerContext& context)
{
    auto tuner = std::make_unique<Tuner>(m_PlatformId, m_DeviceId, m_Api);
    context.SetTuner(std::move(tuner));
}

CommandPriority CreateTunerCommand::GetPriority() const
{
    return CommandPriority::Initialization;
}

} // namespace ktt
