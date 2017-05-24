#pragma once

#include <vector>

#include "../ktt_type_aliases.h"
#include "../enum/argument_data_type.h"
#include "../enum/argument_memory_type.h"
#include "../dto/result_argument.h"

namespace ktt
{

class ManipulatorInterface
{
public:
    // Destructor
    virtual ~ManipulatorInterface() = default;

    // Kernel run methods
    virtual std::vector<ResultArgument> runKernel(const size_t kernelId) = 0;
    virtual std::vector<ResultArgument> runKernel(const size_t kernelId, const DimensionVector& globalSize, const DimensionVector& localSize) = 0;

    // Configuration retrieval methods
    virtual DimensionVector getCurrentGlobalSize(const size_t kernelId) const = 0;
    virtual DimensionVector getCurrentLocalSize(const size_t kernelId) const = 0;
    virtual std::vector<ParameterValue> getCurrentConfiguration() const = 0;

    // Argument update methods
    virtual void updateArgumentScalar(const size_t argumentId, const void* argumentData) = 0;
    virtual void updateArgumentVector(const size_t argumentId, const void* argumentData) = 0;
    virtual void updateArgumentVector(const size_t argumentId, const void* argumentData, const size_t numberOfElements) = 0;
    virtual void setAutomaticArgumentUpdate(const bool flag) = 0;
    virtual void setArgumentSynchronization(const bool flag, const ArgumentMemoryType& argumentMemoryType) = 0;
    virtual void updateKernelArguments(const size_t kernelId, const std::vector<size_t>& argumentIds) = 0;
    virtual void swapKernelArguments(const size_t kernelId, const size_t argumentIdFirst, const size_t argumentIdSecond) = 0;
};

} // namespace ktt
