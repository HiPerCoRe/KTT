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
    case ArgumentDataType::Int:
        return (void*)dataInt.data();
    case ArgumentDataType::Short:
        return (void*)dataShort.data();
    default:
        throw std::runtime_error("Unsupported argument data type");
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

std::vector<short> KernelArgument::getDataShort() const
{
    return dataShort;
}

size_t KernelArgument::getDataSizeInBytes() const
{
    switch (argumentDataType)
    {
    case ArgumentDataType::Double:
        return dataDouble.size() * sizeof(double);
    case ArgumentDataType::Float:
        return dataFloat.size() * sizeof(float);
    case ArgumentDataType::Int:
        return dataInt.size() * sizeof(int);
    case ArgumentDataType::Short:
        return dataShort.size() * sizeof(short);
    default:
        throw std::runtime_error("Unsupported argument data type");
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

bool KernelArgument::operator==(const KernelArgument& other) const
{
    return id == other.id;
}

bool KernelArgument::operator!=(const KernelArgument& other) const
{
    return !(*this == other);
}

std::ostream& operator<<(std::ostream& outputTarget, const KernelArgument& kernelArgument)
{
    if (kernelArgument.argumentDataType == ArgumentDataType::Double)
    {
        printVector(outputTarget, kernelArgument.dataDouble);
    }
    else if (kernelArgument.argumentDataType == ArgumentDataType::Float)
    {
        printVector(outputTarget, kernelArgument.dataFloat);
    }
    else if (kernelArgument.argumentDataType == ArgumentDataType::Int)
    {
        printVector(outputTarget, kernelArgument.dataInt);
    }
    else if (kernelArgument.argumentDataType == ArgumentDataType::Short)
    {
        printVector(outputTarget, kernelArgument.dataShort);
    }
    else
    {
        throw std::runtime_error("Unsupported argument data type was provided for kernel argument");
    }

    return outputTarget;
}

} // namespace ktt
