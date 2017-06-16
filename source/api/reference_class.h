#pragma once

#include "enum/argument_data_type.h"

namespace ktt
{

class ReferenceClass
{
public:
    virtual ~ReferenceClass() = default;
    virtual void computeResult() = 0;
    virtual const void* getData(const size_t argumentId) const = 0;
    virtual size_t getNumberOfElements(const size_t argumentId) const
    {
        return 0;
    }
};

} // namespace ktt
