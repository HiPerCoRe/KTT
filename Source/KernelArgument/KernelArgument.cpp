#include <cstring>

#ifdef KTT_PYTHON
#include <pybind11/stl.h>
#endif // KTT_PYTHON

#include <Api/KttException.h>
#include <KernelArgument/KernelArgument.h>
#include <Python/PythonInterpreter.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/External/half.hpp>
#include <Utility/Logger/Logger.h>
#include <Utility/FileSystem.h>

namespace ktt
{

KernelArgument::KernelArgument(const ArgumentId id, const size_t elementSize, const ArgumentDataType dataType,
    const ArgumentMemoryLocation memoryLocation, const ArgumentAccessType accessType, const ArgumentMemoryType memoryType,
    const ArgumentManagementType managementType, const std::string& symbolName) :
    m_Id(id),
    m_ElementSize(elementSize),
    m_DataSize(0),
    m_DataType(dataType),
    m_MemoryLocation(memoryLocation),
    m_AccessType(accessType),
    m_MemoryType(memoryType),
    m_ManagementType(managementType),
    m_Ownership(ArgumentOwnership::Copy),
    m_SymbolName(symbolName),
    m_ReferencedData(nullptr)
{
    KttAssert(m_MemoryType == ArgumentMemoryType::Vector || m_MemoryLocation == ArgumentMemoryLocation::Undefined,
        "Non-vector arguments must have undefined memory location");
    KttAssert(m_MemoryType != ArgumentMemoryType::Vector || m_MemoryLocation != ArgumentMemoryLocation::Undefined,
        "Vector arguments must have defined memory location");

    if (m_ElementSize == 0)
    {
        throw KttException("Kernel argument element size must be greater than zero");
    }

    if (!m_SymbolName.empty())
    {
        m_SymbolName = "&" + m_SymbolName;
    }
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

    m_Ownership = ArgumentOwnership::Reference;
    m_DataSize = dataSize;
    m_Data.clear();
    m_ReferencedData = data;
}

void KernelArgument::SetOwnedData(const void* data, const size_t dataSize)
{
    if (dataSize == 0 && GetMemoryType() != ArgumentMemoryType::Local)
    {
        throw KttException("Kernel argument cannot be initialized with number of elements equal to zero");
    }

    if (data == nullptr && GetMemoryType() != ArgumentMemoryType::Local)
    {
        throw KttException("Kernel argument cannot be initialized with null data");
    }

    m_Ownership = ArgumentOwnership::Copy;
    m_DataSize = dataSize;
    m_ReferencedData = nullptr;

    if (data != nullptr)
    {
        m_Data.resize(dataSize);
        std::memcpy(m_Data.data(), data, dataSize);
    }
}

void KernelArgument::SetOwnedDataFromFile(const std::string& file)
{
    const auto data = LoadFileToBinary(file);
    SetOwnedData(data.data(), data.size());
}

void KernelArgument::SetOwnedDataFromGenerator([[maybe_unused]] const std::string& generatorFunction, [[maybe_unused]] const size_t dataSize)
{
#ifndef KTT_PYTHON
    throw KttException("Usage of kernel arguments filled from generator function requires compilation of Python backend");
#else
    auto& interpreter = PythonInterpreter::GetInterpreter();
    std::vector<uint8_t> data(dataSize);
    const size_t numberOfElements = dataSize / m_ElementSize;
    pybind11::dict locals;

    for (size_t i = 0; i < numberOfElements; ++i)
    {
        locals["i"] = i;

        try
        {
            switch (m_DataType)
            {
            case ArgumentDataType::Char:
            {
                const auto element = interpreter.Evaluate<int8_t>(generatorFunction, locals);
                std::memcpy(data.data() + m_ElementSize * i, &element, m_ElementSize);
                break;
            }
            case ArgumentDataType::UnsignedChar:
            {
                const auto element = interpreter.Evaluate<uint8_t>(generatorFunction, locals);
                std::memcpy(data.data() + m_ElementSize * i, &element, m_ElementSize);
                break;
            }
            case ArgumentDataType::Short:
            {
                const auto element = interpreter.Evaluate<int16_t>(generatorFunction, locals);
                std::memcpy(data.data() + m_ElementSize * i, &element, m_ElementSize);
                break;
            }
            case ArgumentDataType::UnsignedShort:
            {
                const auto element = interpreter.Evaluate<uint16_t>(generatorFunction, locals);
                std::memcpy(data.data() + m_ElementSize * i, &element, m_ElementSize);
                break;
            }
            case ArgumentDataType::Int:
            {
                const auto element = interpreter.Evaluate<int32_t>(generatorFunction, locals);
                std::memcpy(data.data() + m_ElementSize * i, &element, m_ElementSize);
                break;
            }
            case ArgumentDataType::UnsignedInt:
            {
                const auto element = interpreter.Evaluate<uint32_t>(generatorFunction, locals);
                std::memcpy(data.data() + m_ElementSize * i, &element, m_ElementSize);
                break;
            }
            case ArgumentDataType::Long:
            {
                const auto element = interpreter.Evaluate<int64_t>(generatorFunction, locals);
                std::memcpy(data.data() + m_ElementSize * i, &element, m_ElementSize);
                break;
            }
            case ArgumentDataType::UnsignedLong:
            {
                const auto element = interpreter.Evaluate<uint64_t>(generatorFunction, locals);
                std::memcpy(data.data() + m_ElementSize * i, &element, m_ElementSize);
                break;
            }
            case ArgumentDataType::Half:
            {
                const auto element = interpreter.Evaluate<half_float::half>(generatorFunction, locals);
                std::memcpy(data.data() + m_ElementSize * i, &element, m_ElementSize);
                break;
            }
            case ArgumentDataType::Float:
            {
                const auto element = interpreter.Evaluate<float>(generatorFunction, locals);
                std::memcpy(data.data() + m_ElementSize * i, &element, m_ElementSize);
                break;
            }
            case ArgumentDataType::Double:
            {
                const auto element = interpreter.Evaluate<double>(generatorFunction, locals);
                std::memcpy(data.data() + m_ElementSize * i, &element, m_ElementSize);
                break;
            }
            case ArgumentDataType::Custom:
                throw KttException("Generator functions are not supported for arguments with a custom data type");
            default:
                KttError("Unhandled argument data type");
            }
        }
        catch (const pybind11::error_already_set& exception)
        {
            Logger::LogError(exception.what());
        }
    }

    SetOwnedData(data.data(), data.size());
#endif // KTT_PYTHON
}

void KernelArgument::SetUserBuffer(const size_t dataSize)
{
    if (dataSize == 0)
    {
        throw KttException("Kernel argument cannot be initialized with number of elements equal to zero");
    }

    m_Ownership = ArgumentOwnership::User;
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

ArgumentManagementType KernelArgument::GetManagementType() const
{
    return m_ManagementType;
}

ArgumentOwnership KernelArgument::GetOwnership() const
{
    return m_Ownership;
}

const std::string& KernelArgument::GetSymbolName() const
{
    return m_SymbolName;
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
    switch (m_Ownership)
    {
    case ktt::ArgumentOwnership::Copy:
        return m_Data.data();
    case ktt::ArgumentOwnership::Reference:
        return m_ReferencedData;
    case ktt::ArgumentOwnership::User:
        KttError("Data cannot be retrieved for user argument type");
        return nullptr;
    default:
        KttError("Unhandled argument ownership value");
        return nullptr;
    }
}

void* KernelArgument::GetData()
{
    return const_cast<void*>(static_cast<const KernelArgument*>(this)->GetData());
}

void KernelArgument::SaveData(const std::string& file) const
{
    if (m_MemoryType != ArgumentMemoryType::Vector)
    {
        throw KttException("Only vector arguments can be saved into a file");
    }

    if (HasUserBuffer())
    {
        throw KttException("User-owned vector arguments cannot be saved into a file");
    }

    SaveBinaryToFile(file, GetData(), GetDataSize());
}

bool KernelArgument::HasOwnedData() const
{
    return m_Ownership == ArgumentOwnership::Copy;
}

bool KernelArgument::HasUserBuffer() const
{
    return m_Ownership == ArgumentOwnership::User;
}

std::vector<KernelArgument*> KernelArgument::GetArgumentsWithMemoryType(const std::vector<KernelArgument*>& arguments,
    const ArgumentMemoryType type)
{
    std::vector<KernelArgument*> result;

    for (auto* argument : arguments)
    {
        if (argument->GetMemoryType() == type)
        {
            result.push_back(argument);
        }
    }

    return result;
}

} // namespace ktt
