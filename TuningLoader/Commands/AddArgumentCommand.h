#pragma once

#include <string>

#include <Deserialization/ArgumentFillType.h>
#include <TunerCommand.h>

namespace ktt
{

class AddArgumentCommand : public TunerCommand
{
public:
    AddArgumentCommand() = default;
    explicit AddArgumentCommand(const ArgumentMemoryType memoryType, const ArgumentDataType dataType, const size_t size,
        const size_t typeSize, const ArgumentAccessType accessType, const ArgumentFillType fillType, const float fillValue);
    explicit AddArgumentCommand(const ArgumentMemoryType memoryType, const ArgumentDataType dataType, const size_t size,
        const size_t typeSize, const ArgumentAccessType accessType, const std::string& dataFile);

    virtual void Execute(TunerContext& context) override;
    virtual CommandPriority GetPriority() const override;

private:
    ArgumentMemoryType m_MemoryType;
    ArgumentDataType m_Type;
    size_t m_Size;
    size_t m_TypeSize;
    ArgumentAccessType m_AccessType;
    ArgumentFillType m_FillType;
    float m_FillValue;
    std::string m_DataFile;

    ArgumentId SubmitScalarArgument(TunerContext& context) const;
    ArgumentId SubmitVectorArgument(TunerContext& context) const;
};

} // namespace ktt
