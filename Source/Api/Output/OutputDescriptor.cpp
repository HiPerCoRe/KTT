#include <Api/Output/OutputDescriptor.h>

namespace ktt
{

OutputDescriptor::OutputDescriptor(const ArgumentId id, void* outputDestination) :
    OutputDescriptor(id, outputDestination, 0)
{}

OutputDescriptor::OutputDescriptor(const ArgumentId id, void* outputDestination, const size_t outputSize) :
    m_ArgumentId(id),
    m_OutputDestination(outputDestination),
    m_OutputSize(outputSize)
{}

ArgumentId OutputDescriptor::GetArgumentId() const
{
    return m_ArgumentId;
}

void* OutputDescriptor::GetOutputDestination() const
{
    return m_OutputDestination;
}

size_t OutputDescriptor::GetOutputSize() const
{
    return m_OutputSize;
}

} // namespace ktt
