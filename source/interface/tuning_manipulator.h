#pragma once

#include "../ktt_type_aliases.h"
#include "manipulator_interface.h"

namespace ktt
{

class TuningManipulator
{
public:
    virtual ~TuningManipulator() = default;
    virtual void launchComputation(const size_t kernelId, const DimensionVector& globalSize, const DimensionVector& localSize,
        const std::vector<ParameterValue>& parameterValues) = 0;

    void setManipulatorInterface(ManipulatorInterface* manipulatorInterface)
    {
        this->manipulatorInterface = manipulatorInterface;
    }

protected:
    ManipulatorInterface* manipulatorInterface;
};

} // namespace ktt
