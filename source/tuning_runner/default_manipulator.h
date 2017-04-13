#pragma once

#include "../interface/tuning_manipulator.h"

namespace ktt
{

class DefaultManipulator : public TuningManipulator
{
public:
    virtual void launchComputation(const size_t kernelId, const DimensionVector& globalSize, const DimensionVector& localSize,
        const std::vector<ParameterValue>& parameterValues) override;
};

} // namespace ktt
