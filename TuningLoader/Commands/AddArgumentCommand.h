#pragma once

#include <ArgumentFillType.h>
#include <TunerCommand.h>

namespace ktt
{

class AddArgumentCommand : public TunerCommand
{
public:
    AddArgumentCommand() = default;
    explicit AddArgumentCommand(const ArgumentMemoryType memoryType, const ArgumentDataType dataType, const size_t size,
        const ArgumentAccessType accessType, const ArgumentFillType fillType, const float fillValue, const size_t order);

    virtual void Execute(TunerContext& context) override;
    virtual CommandPriority GetPriority() const override;

    size_t GetOrder() const;

private:
    ArgumentMemoryType m_MemoryType;
    ArgumentDataType m_Type;
    size_t m_Size;
    ArgumentAccessType m_AccessType;
    ArgumentFillType m_FillType;
    float m_FillValue;
    size_t m_Order;
};

} // namespace ktt