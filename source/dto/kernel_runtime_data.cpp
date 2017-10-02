#include "kernel_runtime_data.h"
#include "utility/ktt_utility.h"

namespace ktt
{

KernelRuntimeData::KernelRuntimeData(const std::string& name, const std::string& source, const DimensionVector& globalSize,
    const DimensionVector& localSize, const std::vector<size_t>& argumentIndices) :
    name(name),
    source(source),
    globalSize(convertDimensionVector(globalSize)),
    localSize(convertDimensionVector(localSize)),
    globalSizeDimensionVector(globalSize),
    localSizeDimensionVector(localSize),
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

std::vector<size_t> KernelRuntimeData::getGlobalSize() const
{
    return globalSize;
}

std::vector<size_t> KernelRuntimeData::getLocalSize() const
{
    return localSize;
}


DimensionVector KernelRuntimeData::getGlobalSizeDimensionVector() const
{
    return globalSizeDimensionVector;
}

DimensionVector KernelRuntimeData::getLocalSizeDimensionVector() const
{
    return localSizeDimensionVector;
}

std::vector<size_t> KernelRuntimeData::getArgumentIndices() const
{
    return argumentIndices;
}

} // namespace ktt
