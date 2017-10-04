#include "argument_output_descriptor.h"

namespace ktt
{

ArgumentOutputDescriptor::ArgumentOutputDescriptor(const size_t argumentId, void* outputDestination, const size_t outputSizeInBytes) :
    argumentId(argumentId),
    outputDestination(outputDestination),
    outputSizeInBytes(outputSizeInBytes)
{}

size_t ArgumentOutputDescriptor::getArgumentId() const
{
    return argumentId;
}

void* ArgumentOutputDescriptor::getOutputDestination() const
{
    return outputDestination;
}

size_t ArgumentOutputDescriptor::getOutputSizeInBytes() const
{
    return outputSizeInBytes;
}

} // namespace ktt
