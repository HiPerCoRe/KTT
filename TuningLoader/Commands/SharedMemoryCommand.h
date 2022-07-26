#pragma once

#include <string>

#include <TunerCommand.h>

namespace ktt
{

class SharedMemoryCommand : public TunerCommand
{
public:
    SharedMemoryCommand() = default;
    explicit SharedMemoryCommand(const size_t memorySize);

    virtual void Execute(TunerContext& context) override;
    virtual CommandPriority GetPriority() const override;

private:
    size_t m_MemorySize;
};

} // namespace ktt
