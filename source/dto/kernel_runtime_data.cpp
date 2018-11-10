#include "kernel_runtime_data.h"

namespace ktt
{

KernelRuntimeData::KernelRuntimeData(const KernelId id, const std::string& name, const std::string& source, const DimensionVector& globalSize,
    const DimensionVector& localSize, const std::vector<ArgumentId>& argumentIds) :
    KernelRuntimeData(id, name, source, globalSize, localSize, argumentIds, std::vector<LocalMemoryModifier>{})
{}

KernelRuntimeData::KernelRuntimeData(const KernelId id, const std::string& name, const std::string& source, const DimensionVector& globalSize,
    const DimensionVector& localSize, const std::vector<ArgumentId>& argumentIds, const std::vector<LocalMemoryModifier>& localMemoryModifiers) :
    id(id),
    name(name),
    source(source),
    globalSize(globalSize.getVector()),
    localSize(localSize.getVector()),
    globalSizeDimensionVector(globalSize),
    localSizeDimensionVector(localSize),
    argumentIds(argumentIds),
    localMemoryModifiers(localMemoryModifiers)
{}

void KernelRuntimeData::setGlobalSize(const DimensionVector& globalSize)
{
    this->globalSizeDimensionVector = globalSize;
    this->globalSize = globalSize.getVector();
}

void KernelRuntimeData::setLocalSize(const DimensionVector& localSize)
{
    this->localSizeDimensionVector = localSize;
    this->localSize = localSize.getVector();
}

void KernelRuntimeData::setArgumentIndices(const std::vector<ArgumentId>& argumentIds)
{
    this->argumentIds = argumentIds;
}

KernelId KernelRuntimeData::getId() const
{
    return id;
}

const std::string& KernelRuntimeData::getName() const
{
    return name;
}

const std::string& KernelRuntimeData::getSource() const
{
    return source;
}

const std::vector<size_t>& KernelRuntimeData::getGlobalSize() const
{
    return globalSize;
}

const std::vector<size_t>& KernelRuntimeData::getLocalSize() const
{
    return localSize;
}

const DimensionVector& KernelRuntimeData::getGlobalSizeDimensionVector() const
{
    return globalSizeDimensionVector;
}

const DimensionVector& KernelRuntimeData::getLocalSizeDimensionVector() const
{
    return localSizeDimensionVector;
}

const std::vector<ArgumentId>& KernelRuntimeData::getArgumentIds() const
{
    return argumentIds;
}

const std::vector<LocalMemoryModifier>& KernelRuntimeData::getLocalMemoryModifiers() const
{
    return localMemoryModifiers;
}

} // namespace ktt
