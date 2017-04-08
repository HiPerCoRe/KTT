#include "kernel_argument.h"

namespace ktt
{

size_t KernelArgument::getId() const
{
    return id;
}

const void* KernelArgument::getData() const
{
    switch (argumentDataType)
    {
    case ArgumentDataType::Double:
        return (void*)dataDouble.data();
    case ArgumentDataType::Float:
        return (void*)dataFloat.data();
    default:
        return (void*)dataInt.data();
    }
}

std::vector<double> KernelArgument::getDataDouble() const
{
    return dataDouble;
}

std::vector<float> KernelArgument::getDataFloat() const
{
    return dataFloat;
}

std::vector<int> KernelArgument::getDataInt() const
{
    return dataInt;
}

size_t KernelArgument::getDataSize() const
{
    switch (argumentDataType)
    {
    case ArgumentDataType::Double:
        return dataDouble.size() * sizeof(double);
    case ArgumentDataType::Float:
        return dataFloat.size() * sizeof(float);
    default:
        return dataInt.size() * sizeof(int);
    }
}

ArgumentDataType KernelArgument::getArgumentDataType() const
{
    return argumentDataType;
}

ArgumentMemoryType KernelArgument::getArgumentMemoryType() const
{
    return argumentMemoryType;
}

ArgumentQuantity KernelArgument::getArgumentQuantity() const
{
    return argumentQuantity;
}

} // namespace ktt
