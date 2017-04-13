#include "result_argument.h"

namespace ktt
{

size_t ResultArgument::getId() const
{
    return id;
}

const void* ResultArgument::getData() const
{
    return data;
}

size_t ResultArgument::getDataSizeInBytes() const
{
    return dataSizeInBytes;
}

ArgumentDataType ResultArgument::getArgumentDataType() const
{
    return argumentDataType;
}

} // namespace ktt
