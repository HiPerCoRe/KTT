#include "kernel_runtime_data.h"

namespace ktt
{

KernelRuntimeData::KernelRuntimeData(const std::string& name, const std::string& source, const DimensionVector& globalSize,
    const DimensionVector& localSize, const std::vector<size_t>& argumentIndices) :
    name(name),
    source(source),
    globalSize(globalSize),
    localSize(localSize),
    argumentIndices(argumentIndices)
{}

void KernelRuntimeData::setArgumentIndices(const std::vector<size_t>& argumentIndices)
{
    this->argumentIndices = argumentIndices;
}

std::string KernelRuntimeData::getName() const
{
    return name;
}

std::string KernelRuntimeData::getSource() const
{
    return source;
}

DimensionVector KernelRuntimeData::getGlobalSize() const
{
    return globalSize;
}

DimensionVector KernelRuntimeData::getLocalSize() const
{
    return localSize;
}

std::vector<size_t> KernelRuntimeData::getArgumentIndices() const
{
    return argumentIndices;
}

} // namespace ktt
