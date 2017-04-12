#pragma once

#include "../ktt_type_aliases.h"
#include "../dto/kernel_run_result.h"

namespace ktt
{

class TuningManipulator
{
public:
    virtual ~TuningManipulator() = default;
    virtual KernelRunResult launchComputation(const DimensionVector& globalSize, const DimensionVector& localSize,
        const std::vector<ParameterValue>& parameterValues) = 0;
};

} // namespace ktt
