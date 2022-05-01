#include <TuningLoader/Commands/AddArgumentCommand.h>

namespace ktt
{

AddArgumentCommand::AddArgumentCommand(const std::string& name) :
    m_Name(name)
{}

void AddArgumentCommand::Execute(TunerContext& context)
{

}

CommandPriority AddArgumentCommand::GetPriority() const
{
    return CommandPriority::ArgumentAddition;
}

} // namespace ktt
