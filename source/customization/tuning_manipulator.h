#pragma once

#include <utility>

#include "../ktt_type_aliases.h"
#include "../enum/thread_size_usage.h"
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
    virtual std::vector<std::pair<size_t, ThreadSizeUsage>> getUtilizedKernelIds() const;

    std::vector<ResultArgument> runKernel(const size_t kernelId);
    std::vector<ResultArgument> runKernel(const size_t kernelId, const DimensionVector& globalSize, const DimensionVector& localSize);
    void updateArgumentScalar(const size_t argumentId, const void* argumentData);
    void updateArgumentVector(const size_t argumentId, const void* argumentData);
    void updateArgumentVector(const size_t argumentId, const void* argumentData, const size_t numberOfElements);

    static std::vector<size_t> convertFromDimensionVector(const DimensionVector& vector);
    static DimensionVector convertToDimensionVector(const std::vector<size_t>& vector);

    friend class TuningRunner;

private:
    ManipulatorInterface* manipulatorInterface;
};

} // namespace ktt
