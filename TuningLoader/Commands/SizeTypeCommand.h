#pragma once

#include <TunerCommand.h>

namespace ktt
{

class SizeTypeCommand : public TunerCommand
{
public:
    SizeTypeCommand() = default;
    explicit SizeTypeCommand(const GlobalSizeType sizeType);

    virtual void Execute(TunerContext& context) override;
    virtual CommandPriority GetPriority() const override;

private:
    GlobalSizeType m_SizeType;
};

} // namespace ktt
