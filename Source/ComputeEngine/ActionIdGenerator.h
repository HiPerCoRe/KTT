#pragma once

#include <KttTypes.h>

namespace ktt
{

class ActionIdGenerator
{
public:
    ActionIdGenerator();

    ComputeActionId GenerateComputeId();
    TransferActionId GenerateTransferId();

private:
    ComputeActionId m_NextComputeId;
    TransferActionId m_NextTransferId;
};

} // namespace ktt
