#include <cstring>

#include "result_argument.h"

namespace ktt
{

ResultArgument::ResultArgument(const size_t id, const void* data, const size_t numberOfElements, const ArgumentDataType& argumentDataType) :
    id(id),
    numberOfElements(numberOfElements),
    argumentDataType(argumentDataType)
{
    initializeData(data, numberOfElements, argumentDataType);
}

size_t ResultArgument::getId() const
{
    return id;
}

size_t ResultArgument::getNumberOfElements() const
{
    return numberOfElements;
}

size_t ResultArgument::getDataSizeInBytes() const
{
    return numberOfElements * getElementSizeInBytes();
}

const void* ResultArgument::getData() const
{
    switch (argumentDataType)
    {
    case ArgumentDataType::Char:
        return (void*)dataChar.data();
    case ArgumentDataType::UnsignedChar:
        return (void*)dataUnsignedChar.data();
    case ArgumentDataType::Short:
        return (void*)dataShort.data();
    case ArgumentDataType::UnsignedShort:
        return (void*)dataUnsignedShort.data();
    case ArgumentDataType::Int:
        return (void*)dataInt.data();
    case ArgumentDataType::UnsignedInt:
        return (void*)dataUnsignedInt.data();
    case ArgumentDataType::Long:
        return (void*)dataLong.data();
    case ArgumentDataType::UnsignedLong:
        return (void*)dataUnsignedLong.data();
    case ArgumentDataType::Half:
        return (void*)dataHalf.data();
    case ArgumentDataType::Float:
        return (void*)dataFloat.data();
    default:
        return (void*)dataDouble.data();
    }
}

void* ResultArgument::getData()
{
    return const_cast<void*>(static_cast<const ResultArgument*>(this)->getData());
}

size_t ResultArgument::getElementSizeInBytes() const
{
    switch (argumentDataType)
    {
    case ArgumentDataType::Char:
        return sizeof(int8_t);
    case ArgumentDataType::UnsignedChar:
        return sizeof(uint8_t);
    case ArgumentDataType::Short:
        return sizeof(int16_t);
    case ArgumentDataType::UnsignedShort:
        return sizeof(uint16_t);
    case ArgumentDataType::Int:
        return sizeof(int32_t);
    case ArgumentDataType::UnsignedInt:
        return sizeof(uint32_t);
    case ArgumentDataType::Long:
        return sizeof(int64_t);
    case ArgumentDataType::UnsignedLong:
        return sizeof(uint64_t);
    case ArgumentDataType::Half:
        return sizeof(half);
    case ArgumentDataType::Float:
        return sizeof(float);
    default:
        return sizeof(double);
    }
}

void ResultArgument::initializeData(const void* data, const size_t numberOfElements, const ArgumentDataType& argumentDataType)
{
    prepareData(numberOfElements, argumentDataType);
    std::memcpy(getData(), data, numberOfElements * getElementSizeInBytes());
}

void ResultArgument::prepareData(const size_t numberOfElements, const ArgumentDataType& argumentDataType)
{
    if (argumentDataType == ArgumentDataType::Char)
    {
        dataChar.resize(numberOfElements);
    }
    else if (argumentDataType == ArgumentDataType::UnsignedChar)
    {
        dataUnsignedChar.resize(numberOfElements);
    }
    else if (argumentDataType == ArgumentDataType::Short)
    {
        dataShort.resize(numberOfElements);
    }
    else if (argumentDataType == ArgumentDataType::UnsignedShort)
    {
        dataUnsignedShort.resize(numberOfElements);
    }
    else if (argumentDataType == ArgumentDataType::Int)
    {
        dataInt.resize(numberOfElements);
    }
    else if (argumentDataType == ArgumentDataType::UnsignedInt)
    {
        dataUnsignedInt.resize(numberOfElements);
    }
    else if (argumentDataType == ArgumentDataType::Long)
    {
        dataLong.resize(numberOfElements);
    }
    else if (argumentDataType == ArgumentDataType::UnsignedLong)
    {
        dataUnsignedLong.resize(numberOfElements);
    }
    else if (argumentDataType == ArgumentDataType::Half)
    {
        dataHalf.resize(numberOfElements);
    }
    else if (argumentDataType == ArgumentDataType::Float)
    {
        dataFloat.resize(numberOfElements);
    }
    else
    {
        dataDouble.resize(numberOfElements);
    }
}

} // namespace ktt
