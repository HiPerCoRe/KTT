#pragma once

#include <cstdint>
#include <iostream>
#include <vector>
#include "half.hpp"
#include "ktt_types.h"
#include "enum/argument_access_type.h"
#include "enum/argument_data_type.h"
#include "enum/argument_memory_location.h"
#include "enum/argument_upload_type.h"

namespace ktt
{

using half_float::half;

class KernelArgument
{
public:
    // Constructors
    explicit KernelArgument(const ArgumentId id, const size_t numberOfElements, const ArgumentDataType& dataType,
        const ArgumentMemoryLocation& memoryLocation, const ArgumentAccessType& accessType, const ArgumentUploadType& uploadType);
    explicit KernelArgument(const ArgumentId id, const void* data, const size_t numberOfElements, const ArgumentDataType& dataType,
        const ArgumentMemoryLocation& memoryLocation, const ArgumentAccessType& accessType, const ArgumentUploadType& uploadType,
        const bool dataOwned);

    // Core methods
    void updateData(const void* data, const size_t numberOfElements);

    // Getters
    ArgumentId getId() const;
    size_t getNumberOfElements() const;
    ArgumentDataType getDataType() const;
    ArgumentMemoryLocation getMemoryLocation() const;
    ArgumentAccessType getAccessType() const;
    ArgumentUploadType getUploadType() const;
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

private:
    // Attributes
    ArgumentId id;
    size_t numberOfElements;
    ArgumentDataType argumentDataType;
    ArgumentMemoryLocation argumentMemoryLocation;
    ArgumentAccessType argumentAccessType;
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
    bool dataOwned;
    const void* referencedData;

    // Helper methods
    void initializeData(const void* data, const size_t numberOfElements, const ArgumentDataType& dataType);
    void prepareData(const size_t numberOfElements, const ArgumentDataType& dataType);
};

} // namespace ktt
