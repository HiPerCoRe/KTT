#pragma once

#include <vector>

#include "../ktt_type_aliases.h"

namespace ktt
{

class PSOSearcher;

class KernelConfiguration
{
public:
    explicit KernelConfiguration(const DimensionVector& globalSize, const DimensionVector& localSize,
        const std::vector<ParameterValue>& parameterValues);
    
    DimensionVector getGlobalSize() const;
    DimensionVector getLocalSize() const;
    std::vector<ParameterValue> getParameterValues() const;

    friend class PSOSearcher;
    friend std::ostream& operator<<(std::ostream& outputTarget, const KernelConfiguration& kernelConfiguration);

private:
    DimensionVector globalSize;
    DimensionVector localSize;
    std::vector<ParameterValue> parameterValues;
};

std::ostream& operator<<(std::ostream& outputTarget, const KernelConfiguration& kernelConfiguration);

} // namespace ktt
