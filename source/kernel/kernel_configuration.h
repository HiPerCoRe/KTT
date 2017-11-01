#pragma once

#include <ostream>
#include <utility>
#include <vector>
#include "ktt_types.h"
#include "api/dimension_vector.h"

namespace ktt
{

class PSOSearcher;

class KernelConfiguration
{
public:
    explicit KernelConfiguration(const DimensionVector& globalSize, const DimensionVector& localSize,
        const std::vector<ParameterPair>& parameterPairs);
    explicit KernelConfiguration(const std::vector<std::pair<KernelId, DimensionVector>>& compositionGlobalSizes,
        const std::vector<std::pair<KernelId, DimensionVector>>& compositionLocalSizes, const std::vector<ParameterPair>& parameterPairs);

    DimensionVector getGlobalSize() const;
    DimensionVector getLocalSize() const;
    DimensionVector getCompositionKernelGlobalSize(const KernelId id) const;
    DimensionVector getCompositionKernelLocalSize(const KernelId id) const;
    std::vector<DimensionVector> getGlobalSizes() const;
    std::vector<DimensionVector> getLocalSizes() const;
    std::vector<ParameterPair> getParameterPairs() const;
    bool isComposite() const;

    friend class PSOSearcher;
    friend std::ostream& operator<<(std::ostream&, const KernelConfiguration&);

private:
    DimensionVector globalSize;
    DimensionVector localSize;
    std::vector<std::pair<KernelId, DimensionVector>> compositionGlobalSizes;
    std::vector<std::pair<KernelId, DimensionVector>> compositionLocalSizes;
    std::vector<ParameterPair> parameterPairs;
    bool compositeConfiguration;
};

std::ostream& operator<<(std::ostream& outputTarget, const KernelConfiguration& configuration);

} // namespace ktt
