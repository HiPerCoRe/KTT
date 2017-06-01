#pragma once

#include <cstdint>
#include <iostream>
#include <vector>

#include "../half.hpp"
using half_float::half;

#include "../enum/argument_data_type.h"
#include "../enum/argument_memory_type.h"
#include "../enum/argument_upload_type.h"
#include "../enum/argument_print_condition.h"

namespace ktt
{

class KernelArgument
{
public:
    // Constructors
    explicit KernelArgument(const size_t id, const size_t numberOfElements, const ArgumentDataType& argumentDataType,
        const ArgumentMemoryType& argumentMemoryType, const ArgumentUploadType& argumentUploadType);
    explicit KernelArgument(const size_t id, const void* data, const size_t numberOfElements, const ArgumentDataType& argumentDataType,
        const ArgumentMemoryType& argumentMemoryType, const ArgumentUploadType& argumentUploadType);

    // Core methods
    void updateData(const void* data, const size_t numberOfElements);

    // Getters
    size_t getId() const;
    size_t getNumberOfElements() const;
    ArgumentDataType getArgumentDataType() const;
    ArgumentMemoryType getArgumentMemoryType() const;
    ArgumentUploadType getArgumentUploadType() const;
    size_t getElementSizeInBytes() const;
    size_t getDataSizeInBytes() const;
    const void* getData() const;
    void* getData();
    std::vector<int8_t> getDataChar() const;
    std::vector<uint8_t> getDataUnsignedChar() const;
    std::vector<int16_t> getDataShort() const;
    std::vector<uint16_t> getDataUnsignedShort() const;
    std::vector<int32_t> getDataInt() const;
    std::vector<uint32_t> getDataUnsignedInt() const;
    std::vector<int64_t> getDataLong() const;
    std::vector<uint64_t> getDataUnsignedLong() const;
    std::vector<half> getDataHalf() const;
    std::vector<float> getDataFloat() const;
    std::vector<double> getDataDouble() const;

    // Operators
    bool operator==(const KernelArgument& other) const;
    bool operator!=(const KernelArgument& other) const;
    friend std::ostream& operator<<(std::ostream&, const KernelArgument&);

private:
    // Attributes
    size_t id;
    size_t numberOfElements;
    ArgumentDataType argumentDataType;
    ArgumentMemoryType argumentMemoryType;
    ArgumentUploadType argumentUploadType;
    std::vector<int8_t> dataChar;
    std::vector<uint8_t> dataUnsignedChar;
    std::vector<int16_t> dataShort;
    std::vector<uint16_t> dataUnsignedShort;
    std::vector<int32_t> dataInt;
    std::vector<uint32_t> dataUnsignedInt;
    std::vector<int64_t> dataLong;
    std::vector<uint64_t> dataUnsignedLong;
    std::vector<half> dataHalf;
    std::vector<float> dataFloat;
    std::vector<double> dataDouble;

    // Helper methods
    void initializeData(const void* data, const size_t numberOfElements, const ArgumentDataType& argumentDataType);
    void prepareData(const size_t numberOfElements, const ArgumentDataType& argumentDataType);
};

std::ostream& operator<<(std::ostream& outputTarget, const KernelArgument& kernelArgument);

template <typename T> void printVector(std::ostream& outputTarget, const std::vector<T>& data)
{
    for (size_t i = 0; i < data.size(); i++)
    {
        outputTarget << i << ": " << data.at(i) << std::endl;
    }
}

} // namespace ktt
