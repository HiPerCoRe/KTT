#pragma once

#include "ktt_platform.h"
#include "ktt_types.h"

namespace ktt
{

class KTT_API ArgumentOutputDescriptor
{
public:
    explicit ArgumentOutputDescriptor(const ArgumentId id, void* outputDestination);
    explicit ArgumentOutputDescriptor(const ArgumentId id, void* outputDestination, const size_t outputSizeInBytes);

    ArgumentId getArgumentId() const;
    void* getOutputDestination() const;
    size_t getOutputSizeInBytes() const;

private:
    ArgumentId argumentId;
    void* outputDestination;
    size_t outputSizeInBytes;
};

} // namespace ktt
