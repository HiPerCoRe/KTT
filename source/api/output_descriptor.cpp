#include <api/output_descriptor.h>

namespace ktt
{

OutputDescriptor::OutputDescriptor(const ArgumentId id, void* outputDestination) :
    OutputDescriptor(id, outputDestination, 0)
{}

OutputDescriptor::OutputDescriptor(const ArgumentId id, void* outputDestination, const size_t outputSizeInBytes) :
    argumentId(id),
    outputDestination(outputDestination),
    outputSizeInBytes(outputSizeInBytes)
{}

ArgumentId OutputDescriptor::getArgumentId() const
{
    return argumentId;
}

void* OutputDescriptor::getOutputDestination() const
{
    return outputDestination;
}

size_t OutputDescriptor::getOutputSizeInBytes() const
{
    return outputSizeInBytes;
}

} // namespace ktt
