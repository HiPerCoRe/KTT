#pragma once

#include "ktt_types.h"

namespace ktt
{

class ReferenceClass
{
public:
    virtual ~ReferenceClass() = default;
    virtual void computeResult() = 0;
    virtual const void* getData(const ArgumentId id) const = 0;
    virtual size_t getNumberOfElements(const ArgumentId id) const
    {
        return 0;
    }
};

} // namespace ktt
