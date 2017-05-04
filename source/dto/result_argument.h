#pragma once

#include <vector>
#include <cstddef>

#include "../enum/argument_data_type.h"

namespace ktt
{

class ResultArgument
{
public:
    explicit ResultArgument(const size_t id, const void* data, const size_t dataSizeInBytes, const ArgumentDataType& argumentDataType) :
        id(id),
        data(data),
        argumentDataType(argumentDataType)
    {}

    size_t getId() const;
    const void* getData() const;
    size_t getDataSizeInBytes() const;
    ArgumentDataType getArgumentDataType() const;

private:
    size_t id;
    const void* data;
    size_t dataSizeInBytes;
    ArgumentDataType argumentDataType;
};

} // namespace ktt
