#include <cstdint>

#include <Commands/SharedMemoryCommand.h>

namespace ktt
{

SharedMemoryCommand::SharedMemoryCommand(const size_t memorySize) :
    m_MemorySize(memorySize)
{}

void SharedMemoryCommand::Execute(TunerContext& context)
{
    const auto id = context.GetTuner().AddArgumentLocal<uint8_t>(m_MemorySize);
    context.GetArguments().push_back(id);
    context.GetTuner().SetArguments(context.GetKernelDefinitionId(), context.GetArguments());
}

CommandPriority SharedMemoryCommand::GetPriority() const
{
    return CommandPriority::SharedMemoryAddition;
}

} // namespace ktt
