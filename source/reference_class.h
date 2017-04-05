#pragma once

#include "enum/argument_data_type.h"

namespace ktt
{

class ReferenceClass
{
public:
    virtual ~ReferenceClass() = default;
    virtual void* getResultData() = 0;
    virtual ArgumentDataType getResultDataType() const = 0;
    virtual size_t getResultSizeInBytes() const = 0;
};

} // namespace ktt
