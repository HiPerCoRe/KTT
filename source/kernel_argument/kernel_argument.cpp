#include <cstring>
#include <stdexcept>

#include "kernel_argument.h"

namespace ktt
{

KernelArgument::KernelArgument(const size_t id, const size_t numberOfElements, const ArgumentDataType& argumentDataType,
    const ArgumentMemoryType& argumentMemoryType, const ArgumentUploadType& argumentUploadType) :
    id(id),
    numberOfElements(numberOfElements),
    argumentDataType(argumentDataType),
    argumentMemoryType(argumentMemoryType),
    argumentUploadType(argumentUploadType)
{
    if (numberOfElements == 0)
    {
        throw std::runtime_error("Data provided for kernel argument is empty");
    }
    prepareData(numberOfElements, argumentDataType);
}

KernelArgument::KernelArgument(const size_t id, const void* data, const size_t numberOfElements, const ArgumentDataType& argumentDataType,
    const ArgumentMemoryType& argumentMemoryType, const ArgumentUploadType& argumentUploadType) :
    id(id),
    numberOfElements(numberOfElements),
    argumentDataType(argumentDataType),
    argumentMemoryType(argumentMemoryType),
    argumentUploadType(argumentUploadType)
{
    if (numberOfElements == 0)
    {
        throw std::runtime_error("Data provided for kernel argument is empty");
    }

    if (data != nullptr)
    {
        initializeData(data, numberOfElements, argumentDataType);
    }
}

void KernelArgument::updateData(const void* data, const size_t numberOfElements)
{
    if (numberOfElements == 0)
    {
        throw std::runtime_error("Data provided for kernel argument is empty");
    }
    this->numberOfElements = numberOfElements;
    initializeData(data, numberOfElements, argumentDataType);
}

size_t KernelArgument::getId() const
{
    return id;
}

size_t KernelArgument::getNumberOfElements() const
{
    return numberOfElements;
}

ArgumentDataType KernelArgument::getArgumentDataType() const
{
    return argumentDataType;
}

ArgumentMemoryType KernelArgument::getArgumentMemoryType() const
{
    return argumentMemoryType;
}

ArgumentUploadType KernelArgument::getArgumentUploadType() const
{
    return argumentUploadType;
}

size_t KernelArgument::getElementSizeInBytes() const
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
    case ArgumentDataType::Double:
        return sizeof(double);
    default:
        throw std::runtime_error("Unsupported argument data type");
    }
}

size_t KernelArgument::getDataSizeInBytes() const
{
    return numberOfElements * getElementSizeInBytes();
}

const void* KernelArgument::getData() const
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
    case ArgumentDataType::Double:
        return (void*)dataDouble.data();
    default:
        throw std::runtime_error("Unsupported argument data type");
    }
}

void* KernelArgument::getData()
{
    return const_cast<void*>(static_cast<const KernelArgument*>(this)->getData());
}

std::vector<int8_t> KernelArgument::getDataChar() const
{
    return dataChar;
}

std::vector<uint8_t> KernelArgument::getDataUnsignedChar() const
{
    return dataUnsignedChar;
}

std::vector<int16_t> KernelArgument::getDataShort() const
{
    return dataShort;
}

std::vector<uint16_t> KernelArgument::getDataUnsignedShort() const
{
    return dataUnsignedShort;
}

std::vector<int32_t> KernelArgument::getDataInt() const
{
    return dataInt;
}

std::vector<uint32_t> KernelArgument::getDataUnsignedInt() const
{
    return dataUnsignedInt;
}

std::vector<int64_t> KernelArgument::getDataLong() const
{
    return dataLong;
}

std::vector<uint64_t> KernelArgument::getDataUnsignedLong() const
{
    return dataUnsignedLong;
}

std::vector<half> KernelArgument::getDataHalf() const
{
    return dataHalf;
}

std::vector<float> KernelArgument::getDataFloat() const
{
    return dataFloat;
}

std::vector<double> KernelArgument::getDataDouble() const
{
    return dataDouble;
}

bool KernelArgument::operator==(const KernelArgument& other) const
{
    return id == other.id;
}

bool KernelArgument::operator!=(const KernelArgument& other) const
{
    return !(*this == other);
}

void KernelArgument::initializeData(const void* data, const size_t numberOfElements, const ArgumentDataType& argumentDataType)
{
    prepareData(numberOfElements, argumentDataType);
    std::memcpy(getData(), data, numberOfElements * getElementSizeInBytes());
}

void KernelArgument::prepareData(const size_t numberOfElements, const ArgumentDataType& argumentDataType)
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
    else if (argumentDataType == ArgumentDataType::Double)
    {
        dataDouble.resize(numberOfElements);
    }
    else
    {
        throw std::runtime_error("Unsupported argument data type was provided for kernel argument");
    }
}

std::ostream& operator<<(std::ostream& outputTarget, const KernelArgument& kernelArgument)
{
    if (kernelArgument.argumentDataType == ArgumentDataType::Char)
    {
        printVector(outputTarget, kernelArgument.dataChar);
    }
    else if (kernelArgument.argumentDataType == ArgumentDataType::UnsignedChar)
    {
        printVector(outputTarget, kernelArgument.dataUnsignedChar);
    }
    else if (kernelArgument.argumentDataType == ArgumentDataType::Short)
    {
        printVector(outputTarget, kernelArgument.dataShort);
    }
    else if (kernelArgument.argumentDataType == ArgumentDataType::UnsignedShort)
    {
        printVector(outputTarget, kernelArgument.dataUnsignedShort);
    }
    else if (kernelArgument.argumentDataType == ArgumentDataType::Int)
    {
        printVector(outputTarget, kernelArgument.dataInt);
    }
    else if (kernelArgument.argumentDataType == ArgumentDataType::UnsignedInt)
    {
        printVector(outputTarget, kernelArgument.dataUnsignedInt);
    }
    else if (kernelArgument.argumentDataType == ArgumentDataType::Long)
    {
        printVector(outputTarget, kernelArgument.dataLong);
    }
    else if (kernelArgument.argumentDataType == ArgumentDataType::UnsignedLong)
    {
        printVector(outputTarget, kernelArgument.dataUnsignedLong);
    }
    else if (kernelArgument.argumentDataType == ArgumentDataType::Half)
    {
        printVector(outputTarget, kernelArgument.dataHalf);
    }
    else if (kernelArgument.argumentDataType == ArgumentDataType::Float)
    {
        printVector(outputTarget, kernelArgument.dataFloat);
    }
    else if (kernelArgument.argumentDataType == ArgumentDataType::Double)
    {
        printVector(outputTarget, kernelArgument.dataDouble);
    }
    else
    {
        throw std::runtime_error("Unsupported argument data type was provided for kernel argument");
    }

    return outputTarget;
}

} // namespace ktt
