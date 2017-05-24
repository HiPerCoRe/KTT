#pragma once

#include <cstddef>
#include <vector>

#include "../enum/argument_data_type.h"
#include "../enum/argument_memory_type.h"

namespace ktt
{

class ResultArgument
{
public:
    explicit ResultArgument(const size_t id, const void* data, const size_t numberOfElements, const size_t elementSizeInBytes,
        const ArgumentDataType& argumentDataType, const ArgumentMemoryType& argumentMemoryType);

    size_t getId() const;
    size_t getNumberOfElements() const;
    size_t getElementSizeInBytes() const;
    size_t getDataSizeInBytes() const;
    ArgumentDataType getArgumentDataType() const;
    ArgumentMemoryType getArgumentMemoryType() const;
    const void* getData() const;

private:
    size_t id;
    size_t numberOfElements;
    size_t elementSizeInBytes;
    ArgumentDataType argumentDataType;
    ArgumentMemoryType argumentMemoryType;
    const void* data;
};

} // namespace ktt
