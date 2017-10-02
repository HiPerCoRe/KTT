#pragma once

namespace ktt
{

class ArgumentOutputDescriptor
{
public:
    ArgumentOutputDescriptor(const size_t argumentId, void* outputDestination, const size_t outputSizeInBytes);

    size_t getArgumentId() const;
    void* getOutputDestination() const;
    size_t getOutputSizeInBytes() const;

private:
    size_t argumentId;
    void* outputDestination;
    size_t outputSizeInBytes;
};

} // namespace ktt
