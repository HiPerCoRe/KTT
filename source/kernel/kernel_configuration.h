#pragma once

#include <vector>

#include "../ktt_aliases.h"

namespace ktt
{

class KernelConfiguration
{
public:
    explicit KernelConfiguration(const DimensionVector& globalSize, const DimensionVector& localSize,
        const std::vector<ParameterValue>& parameterValues):
        globalSize(globalSize),
        localSize(localSize),
        parameterValues(parameterValues)
    {}
    
    DimensionVector getGlobalSize() const
    {
        return globalSize;
    }

    DimensionVector getLocalSize() const
    {
        return localSize;
    }

    std::vector<ParameterValue> getParameterValues() const
    {
        return parameterValues;
    }

private:
    DimensionVector globalSize;
    DimensionVector localSize;
    std::vector<ParameterValue> parameterValues;
};

} // namespace ktt
