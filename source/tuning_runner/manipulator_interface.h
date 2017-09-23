#pragma once

#include <cstddef>
#include <vector>

#include "ktt_type_aliases.h"
#include "api/result_argument.h"

namespace ktt
{

class ManipulatorInterface
{
public:
    // Destructor
    virtual ~ManipulatorInterface() = default;

    // Kernel run methods
    virtual void runKernel(const size_t kernelId) = 0;
    virtual void runKernel(const size_t kernelId, const DimensionVector& globalSize, const DimensionVector& localSize) = 0;

    // Configuration retrieval methods
    virtual DimensionVector getCurrentGlobalSize(const size_t kernelId) const = 0;
    virtual DimensionVector getCurrentLocalSize(const size_t kernelId) const = 0;
    virtual std::vector<ParameterValue> getCurrentConfiguration() const = 0;

    // Argument update and retrieval methods
    virtual void updateArgumentScalar(const size_t argumentId, const void* argumentData) = 0;
    virtual void updateArgumentLocal(const size_t argumentId, const size_t numberOfElements) = 0;
    virtual void updateArgumentVector(const size_t argumentId, const void* argumentData) = 0;
    virtual void updateArgumentVector(const size_t argumentId, const void* argumentData, const size_t numberOfElements) = 0;
    virtual ResultArgument getArgumentVector(const size_t argumentId) = 0;

    // Kernel argument handling methods
    virtual void changeKernelArguments(const size_t kernelId, const std::vector<size_t>& argumentIds) = 0;
    virtual void swapKernelArguments(const size_t kernelId, const size_t argumentIdFirst, const size_t argumentIdSecond) = 0;
};

} // namespace ktt
