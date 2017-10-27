#pragma once

#include <cstddef>
#include <vector>
#include "ktt_types.h"
#include "api/dimension_vector.h"

namespace ktt
{

class ManipulatorInterface
{
public:
    // Destructor
    virtual ~ManipulatorInterface() = default;

    // Kernel run methods
    virtual void runKernel(const KernelId id) = 0;
    virtual void runKernel(const KernelId id, const DimensionVector& globalSize, const DimensionVector& localSize) = 0;

    // Configuration retrieval methods
    virtual DimensionVector getCurrentGlobalSize(const KernelId id) const = 0;
    virtual DimensionVector getCurrentLocalSize(const KernelId id) const = 0;
    virtual std::vector<ParameterPair> getCurrentConfiguration() const = 0;

    // Argument update and retrieval methods
    virtual void updateArgumentScalar(const ArgumentId id, const void* argumentData) = 0;
    virtual void updateArgumentLocal(const ArgumentId id, const size_t numberOfElements) = 0;
    virtual void updateArgumentVector(const ArgumentId id, const void* argumentData) = 0;
    virtual void updateArgumentVector(const ArgumentId id, const void* argumentData, const size_t numberOfElements) = 0;
    virtual void getArgumentVector(const ArgumentId id, void* destination) const = 0;
    virtual void getArgumentVector(const ArgumentId id, void* destination, const size_t numberOfElements) const = 0;

    // Kernel argument handling methods
    virtual void changeKernelArguments(const KernelId id, const std::vector<ArgumentId>& argumentIds) = 0;
    virtual void swapKernelArguments(const KernelId id, const ArgumentId argumentIdFirst, const ArgumentId argumentIdSecond) = 0;

    // Buffer handling methods
    virtual void createArgumentBuffer(const ArgumentId id) = 0;
    virtual void destroyArgumentBuffer(const ArgumentId id) = 0;
};

} // namespace ktt
