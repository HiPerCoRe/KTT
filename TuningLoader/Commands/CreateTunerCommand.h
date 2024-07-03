#pragma once

#include <TunerCommand.h>
#include <KttTypes.h>

namespace ktt
{

class CreateTunerCommand : public TunerCommand
{
public:
    CreateTunerCommand() = default;
    explicit CreateTunerCommand(const PlatformIndex platformId, const DeviceIndex deviceId, const ComputeApi api);

    virtual void Execute(TunerContext& context) override;
    virtual CommandPriority GetPriority() const override;

private:
    PlatformIndex m_PlatformId;
    DeviceIndex m_DeviceId;
    ComputeApi m_Api;
};

} // namespace ktt
