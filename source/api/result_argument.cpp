#include "result_argument.h"

namespace ktt
{

ResultArgument::ResultArgument(const size_t id, const void* data, const size_t numberOfElements, const size_t elementSizeInBytes,
    const ArgumentDataType& argumentDataType, const ArgumentMemoryType& argumentMemoryType) :
    id(id),
    data(data),
    numberOfElements(numberOfElements),
    elementSizeInBytes(elementSizeInBytes),
    argumentDataType(argumentDataType),
    argumentMemoryType(argumentMemoryType)
{}

size_t ResultArgument::getId() const
{
    return id;
}

size_t ResultArgument::getNumberOfElements() const
{
    return numberOfElements;
}

size_t ResultArgument::getElementSizeInBytes() const
{
    return elementSizeInBytes;
}

size_t ResultArgument::getDataSizeInBytes() const
{
    return numberOfElements * elementSizeInBytes;
}

ArgumentDataType ResultArgument::getArgumentDataType() const
{
    return argumentDataType;
}

ArgumentMemoryType ResultArgument::getArgumentMemoryType() const
{
    return argumentMemoryType;
}

const void* ResultArgument::getData() const
{
    return data;
}

} // namespace ktt
