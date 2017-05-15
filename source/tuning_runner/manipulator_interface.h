#pragma once

#include <vector>

#include "../ktt_type_aliases.h"
#include "../enum/argument_data_type.h"
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

    // Argument update methods
    virtual void updateArgumentScalar(const size_t argumentId, const void* argumentData) = 0;
    virtual void updateArgumentVector(const size_t argumentId, const void* argumentData) = 0;
    virtual void updateArgumentVector(const size_t argumentId, const void* argumentData, const size_t numberOfElements) = 0;
};

} // namespace ktt
