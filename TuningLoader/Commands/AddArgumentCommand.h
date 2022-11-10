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
    explicit AddArgumentCommand(const ArgumentId& id, const ArgumentMemoryType memoryType, const ArgumentDataType dataType,
        const size_t elementCount, const size_t typeSize, const ArgumentAccessType accessType, const ArgumentFillType fillType,
        const float fillValue);
    explicit AddArgumentCommand(const ArgumentId& id, const ArgumentMemoryType memoryType, const ArgumentDataType dataType,
        const size_t elementCount, const size_t typeSize, const ArgumentAccessType accessType, const ArgumentFillType fillType,
        const std::string& dataSource);

    virtual void Execute(TunerContext& context) override;
    virtual CommandPriority GetPriority() const override;

    void SetReferenceProperties(const AddArgumentCommand& other);
    const ArgumentId& GetId() const;

private:
    ArgumentId m_Id;
    ArgumentMemoryType m_MemoryType;
    ArgumentDataType m_Type;
    size_t m_ElementCount;
    size_t m_ElementSize;
    ArgumentAccessType m_AccessType;
    ArgumentFillType m_FillType;
    float m_FillValue;
    std::string m_DataSource;
    bool m_IsReference;

    ArgumentId SubmitScalarArgument(TunerContext& context) const;
    ArgumentId SubmitVectorArgument(TunerContext& context) const;
};

} // namespace ktt
