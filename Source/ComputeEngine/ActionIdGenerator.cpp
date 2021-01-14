#include <ComputeEngine/ActionIdGenerator.h>

namespace ktt
{

ActionIdGenerator::ActionIdGenerator() :
    m_NextComputeId(0),
    m_NextTransferId(0)
{}

ComputeActionId ActionIdGenerator::GenerateComputeId()
{
    return m_NextComputeId++;
}

TransferActionId ActionIdGenerator::GenerateTransferId()
{
    return m_NextTransferId++;
}

} // namespace ktt
