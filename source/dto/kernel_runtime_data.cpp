#include "kernel_runtime_data.h"
#include "utility/ktt_utility.h"

namespace ktt
{

KernelRuntimeData::KernelRuntimeData(const size_t id, const std::string& name, const std::string& source, const DimensionVector& globalSize,
    const DimensionVector& localSize, const std::vector<size_t>& argumentIndices) :
    id(id),
    name(name),
    source(source),
    globalSize(convertDimensionVector(globalSize)),
    localSize(convertDimensionVector(localSize)),
    globalSizeDimensionVector(globalSize),
    localSizeDimensionVector(localSize),
    argumentIndices(argumentIndices)
{}

void KernelRuntimeData::setGlobalSize(const DimensionVector& globalSize)
{
    this->globalSizeDimensionVector = globalSize;
    this->globalSize = convertDimensionVector(globalSize);
}

void KernelRuntimeData::setLocalSize(const DimensionVector& localSize)
{
    this->localSizeDimensionVector = localSize;
    this->localSize = convertDimensionVector(localSize);
}

void KernelRuntimeData::setArgumentIndices(const std::vector<size_t>& argumentIndices)
{
    this->argumentIndices = argumentIndices;
}

size_t KernelRuntimeData::getId() const
{
    return id;
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
