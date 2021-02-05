#include <cstring>

#include <KernelArgument/KernelArgument.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/ErrorHandling/KttException.h>

namespace ktt
{

KernelArgument::KernelArgument(const ArgumentId id, const size_t elementSize, const ArgumentDataType dataType,
    const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const ArgumentMemoryType memoryType) :
    m_Id(id),
    m_ElementSize(elementSize),
    m_DataSize(0),
    m_DataType(dataType),
    m_MemoryLocation(memoryLocation),
    m_AccessType(accessType),
    m_MemoryType(memoryType),
    m_Type(ArgumentType::Copy),
    m_ReferencedData(nullptr)
{
    KttAssert(m_MemoryType == ArgumentMemoryType::Vector || m_MemoryLocation == ArgumentMemoryLocation::Undefined,
        "Non-vector arguments must have undefined memory location");
    KttAssert(m_MemoryType != ArgumentMemoryType::Vector || m_MemoryLocation != ArgumentMemoryLocation::Undefined,
        "Vector arguments must have defined memory location");
}

void KernelArgument::SetReferencedData(void* data, const size_t dataSize)
{
    if (dataSize == 0)
    {
        throw KttException("Kernel argument cannot be initialized with number of elements equal to zero");
    }

    if (data == nullptr)
    {
        throw KttException("Kernel argument cannot be initialized with null data");
    }

    m_Type = ArgumentType::Reference;
    m_DataSize = dataSize;
    m_Data.clear();
    m_ReferencedData = data;
}

void KernelArgument::SetOwnedData(const void* data, const size_t dataSize)
{
    if (dataSize == 0)
    {
        throw KttException("Kernel argument cannot be initialized with number of elements equal to zero");
    }

    if (data == nullptr && GetMemoryType() != ArgumentMemoryType::Local)
    {
        throw KttException("Kernel argument cannot be initialized with null data");
    }

    m_Type = ArgumentType::Copy;
    m_DataSize = dataSize;
    m_ReferencedData = nullptr;

    if (data != nullptr)
    {
        m_Data.resize(dataSize);
        std::memcpy(m_Data.data(), data, dataSize);
    }
}

void KernelArgument::SetUserBuffer(const size_t dataSize)
{
    if (dataSize == 0)
    {
        throw KttException("Kernel argument cannot be initialized with number of elements equal to zero");
    }

    m_Type = ArgumentType::User;
    m_DataSize = dataSize;
    m_Data.clear();
    m_ReferencedData = nullptr;
}

ArgumentId KernelArgument::GetId() const
{
    return m_Id;
}

size_t KernelArgument::GetElementSize() const
{
    return m_ElementSize;
}

ArgumentDataType KernelArgument::GetDataType() const
{
    return m_DataType;
}

ArgumentMemoryLocation KernelArgument::GetMemoryLocation() const
{
    return m_MemoryLocation;
}

ArgumentAccessType KernelArgument::GetAccessType() const
{
    return m_AccessType;
}

ArgumentMemoryType KernelArgument::GetMemoryType() const
{
    return m_MemoryType;
}

uint64_t KernelArgument::GetNumberOfElements() const
{
    return static_cast<uint64_t>(GetDataSize() / GetElementSize());
}

size_t KernelArgument::GetDataSize() const
{
    return m_DataSize;
}

const void* KernelArgument::GetData() const
{
    switch (m_Type)
    {
    case ktt::ArgumentType::Copy:
        return m_Data.data();
    case ktt::ArgumentType::Reference:
        return m_ReferencedData;
    case ktt::ArgumentType::User:
        KttError("Data cannot be retrieved for user argument type");
        return nullptr;
    default:
        KttError("Unhandled argument type value");
        return nullptr;
    }
}

void* KernelArgument::GetData()
{
    return const_cast<void*>(static_cast<const KernelArgument*>(this)->GetData());
}

bool KernelArgument::HasOwnedData() const
{
    return m_Type == ArgumentType::Copy;
}

bool KernelArgument::HasUserBuffer() const
{
    return m_Type == ArgumentType::User;
}

} // namespace ktt
