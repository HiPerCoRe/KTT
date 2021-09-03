#include <type_traits>

#include <Tuner.h>

namespace ktt
{

using half_float::half;

template <typename T>
ArgumentId Tuner::AddArgumentVector(const std::vector<T>& data, const ArgumentAccessType accessType)
{
    const size_t elementSize = sizeof(T);
    const ArgumentDataType dataType = DeriveArgumentDataType<T>();

    return AddArgumentWithOwnedData(elementSize, dataType, ArgumentMemoryLocation::Device, accessType, ArgumentMemoryType::Vector,
        ArgumentManagementType::Framework, data.data(), data.size() * elementSize);
}

template <typename T>
ArgumentId Tuner::AddArgumentVector(std::vector<T>& data, const ArgumentAccessType accessType,
    const ArgumentMemoryLocation memoryLocation, const ArgumentManagementType managementType, const bool referenceUserData)
{
    const size_t elementSize = sizeof(T);
    const size_t dataSize = data.size() * elementSize;
    const ArgumentDataType dataType = DeriveArgumentDataType<T>();

    if (referenceUserData)
    {
        return AddArgumentWithReferencedData(elementSize, dataType, memoryLocation, accessType, ArgumentMemoryType::Vector,
            managementType, data.data(), dataSize);
    }

    return AddArgumentWithOwnedData(elementSize, dataType, memoryLocation, accessType, ArgumentMemoryType::Vector, managementType,
        data.data(), dataSize);
}

template <typename T>
ArgumentId Tuner::AddArgumentVector(ComputeBuffer buffer, const size_t bufferSize, const ArgumentAccessType accessType,
    const ArgumentMemoryLocation memoryLocation)
{
    const size_t elementSize = sizeof(T);
    const ArgumentDataType dataType = DeriveArgumentDataType<T>();

    return AddUserArgument(buffer, elementSize, dataType, memoryLocation, accessType, bufferSize);
}

template <typename T>
ArgumentId Tuner::AddArgumentScalar(const T& data)
{
    const ArgumentDataType dataType = DeriveArgumentDataType<T>();
    return AddArgumentWithOwnedData(sizeof(T), dataType, ArgumentMemoryLocation::Undefined, ArgumentAccessType::ReadOnly,
        ArgumentMemoryType::Scalar, ArgumentManagementType::Framework, &data, sizeof(T));
}

template <typename T>
ArgumentId Tuner::AddArgumentLocal(const size_t localMemorySize)
{
    const ArgumentDataType dataType = DeriveArgumentDataType<T>();
    return AddArgumentWithOwnedData(sizeof(T), dataType, ArgumentMemoryLocation::Undefined, ArgumentAccessType::ReadOnly,
        ArgumentMemoryType::Local, ArgumentManagementType::Framework, nullptr, localMemorySize);
}

template <typename T>
ArgumentId Tuner::AddArgumentSymbol(const T& data, const std::string& symbolName)
{
    const ArgumentDataType dataType = DeriveArgumentDataType<T>();
    return AddArgumentWithOwnedData(sizeof(T), dataType, ArgumentMemoryLocation::Undefined, ArgumentAccessType::ReadOnly,
        ArgumentMemoryType::Symbol, ArgumentManagementType::Framework, &data, sizeof(T), symbolName);
}

template <typename T>
ArgumentDataType Tuner::DeriveArgumentDataType() const
{
    static_assert(std::is_trivially_copyable_v<T> && !std::is_reference_v<T> && !std::is_pointer_v<T> && !std::is_null_pointer_v<T>,
        "Unsupported argument data type");
    static_assert(!std::is_same_v<std::remove_cv_t<T>, bool>, "Bool argument data type is not supported");

    if constexpr (std::is_same_v<std::remove_cv_t<T>, half>)
    {
        return ArgumentDataType::Half;
    }
    else if constexpr (!std::is_arithmetic_v<T>)
    {
        return ArgumentDataType::Custom;
    }
    else if constexpr (sizeof(T) == 1 && std::is_unsigned_v<T>)
    {
        return ArgumentDataType::UnsignedChar;
    }
    else if constexpr (sizeof(T) == 1)
    {
        return ArgumentDataType::Char;
    }
    else if constexpr (sizeof(T) == 2 && std::is_unsigned_v<T>)
    {
        return ArgumentDataType::UnsignedShort;
    }
    else if constexpr (sizeof(T) == 2)
    {
        return ArgumentDataType::Short;
    }
    else if constexpr (std::is_same_v<std::remove_cv_t<T>, float>)
    {
        return ArgumentDataType::Float;
    }
    else if constexpr (sizeof(T) == 4 && std::is_unsigned_v<T>)
    {
        return ArgumentDataType::UnsignedInt;
    }
    else if constexpr (sizeof(T) == 4)
    {
        return ArgumentDataType::Int;
    }
    else if constexpr (std::is_same_v<std::remove_cv_t<T>, double>)
    {
        return ArgumentDataType::Double;
    }
    else if constexpr (sizeof(T) == 8 && std::is_unsigned_v<T>)
    {
        return ArgumentDataType::UnsignedLong;
    }
    else if constexpr (sizeof(T) == 8)
    {
        return ArgumentDataType::Long;
    }
    else
    {
        // Unknown arithmetic type
        return ArgumentDataType::Custom;
    }
}

} // namespace ktt
