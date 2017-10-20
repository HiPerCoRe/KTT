#include "argument_output_descriptor.h"

namespace ktt
{

ArgumentOutputDescriptor::ArgumentOutputDescriptor(const ArgumentId id, void* outputDestination) :
    ArgumentOutputDescriptor(id, outputDestination, 0)
{}

ArgumentOutputDescriptor::ArgumentOutputDescriptor(const ArgumentId id, void* outputDestination, const size_t outputSizeInBytes) :
    argumentId(id),
    outputDestination(outputDestination),
    outputSizeInBytes(outputSizeInBytes)
{}

ArgumentId ArgumentOutputDescriptor::getArgumentId() const
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
