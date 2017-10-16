#pragma once

#include <ostream>
#include <utility>
#include <vector>

#include "ktt_type_aliases.h"

namespace ktt
{

class PSOSearcher;

class KernelConfiguration
{
public:
    explicit KernelConfiguration(const DimensionVector& globalSize, const DimensionVector& localSize,
        const std::vector<ParameterValue>& parameterValues);
    explicit KernelConfiguration(const std::vector<std::pair<size_t, DimensionVector>>& globalSizes,
        const std::vector<std::pair<size_t, DimensionVector>>& localSizes, const std::vector<ParameterValue>& parameterValues);

    DimensionVector getGlobalSize() const;
    DimensionVector getLocalSize() const;
    DimensionVector getGlobalSize(const size_t kernelId) const;
    DimensionVector getLocalSize(const size_t kernelId) const;
    std::vector<DimensionVector> getGlobalSizes() const;
    std::vector<DimensionVector> getLocalSizes() const;
    std::vector<ParameterValue> getParameterValues() const;

    friend class PSOSearcher;
    friend std::ostream& operator<<(std::ostream&, const KernelConfiguration&);

private:
    DimensionVector globalSize;
    DimensionVector localSize;
    std::vector<std::pair<size_t, DimensionVector>> globalSizes;
    std::vector<std::pair<size_t, DimensionVector>> localSizes;
    std::vector<ParameterValue> parameterValues;
};

std::ostream& operator<<(std::ostream& outputTarget, const KernelConfiguration& kernelConfiguration);

} // namespace ktt
