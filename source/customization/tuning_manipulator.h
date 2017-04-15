#pragma once

#include "../ktt_type_aliases.h"
#include "../tuning_runner/manipulator_interface.h"

namespace ktt
{

class TuningRunner;

class TuningManipulator
{
public:
    virtual ~TuningManipulator();
    virtual void launchComputation(const size_t kernelId, const DimensionVector& globalSize, const DimensionVector& localSize,
        const std::vector<ParameterValue>& parameterValues) = 0;

    std::vector<ResultArgument> runKernel(const size_t kernelId);
    std::vector<ResultArgument> runKernel(const size_t kernelId, const DimensionVector& globalSize, const DimensionVector& localSize);
    void updateArgumentScalar(const size_t argumentId, const void* argumentData);
    void updateArgumentVector(const size_t argumentId, const void* argumentData, const size_t dataSizeInBytes);

    friend class TuningRunner;

private:
    ManipulatorInterface* manipulatorInterface;

    void setManipulatorInterface(ManipulatorInterface* manipulatorInterface);
};

} // namespace ktt
