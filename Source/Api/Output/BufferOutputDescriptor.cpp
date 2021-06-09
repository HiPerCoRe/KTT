#include <Api/Output/BufferOutputDescriptor.h>

namespace ktt
{

BufferOutputDescriptor::BufferOutputDescriptor(const ArgumentId id, void* outputDestination) :
    BufferOutputDescriptor(id, outputDestination, 0)
{}

BufferOutputDescriptor::BufferOutputDescriptor(const ArgumentId id, void* outputDestination, const size_t outputSize) :
    m_ArgumentId(id),
    m_OutputDestination(outputDestination),
    m_OutputSize(outputSize)
{}

ArgumentId BufferOutputDescriptor::GetArgumentId() const
{
    return m_ArgumentId;
}

void* BufferOutputDescriptor::GetOutputDestination() const
{
    return m_OutputDestination;
}

size_t BufferOutputDescriptor::GetOutputSize() const
{
    return m_OutputSize;
}

} // namespace ktt
