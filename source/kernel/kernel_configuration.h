#pragma once

#include <ostream>
#include <utility>
#include <vector>
#include "ktt_types.h"
#include "api/dimension_vector.h"
#include "api/parameter_pair.h"
#include "dto/local_memory_modifier.h"

namespace ktt
{

class PSOSearcher;

class KernelConfiguration
{
public:
    KernelConfiguration();
    explicit KernelConfiguration(const DimensionVector& globalSize, const DimensionVector& localSize,
        const std::vector<ParameterPair>& parameterPairs);
    explicit KernelConfiguration(const DimensionVector& globalSize, const DimensionVector& localSize,
        const std::vector<ParameterPair>& parameterPairs, const std::vector<LocalMemoryModifier>& localMemoryModifiers);
    explicit KernelConfiguration(const std::vector<std::pair<KernelId, DimensionVector>>& compositionGlobalSizes,
        const std::vector<std::pair<KernelId, DimensionVector>>& compositionLocalSizes, const std::vector<ParameterPair>& parameterPairs);
    explicit KernelConfiguration(const std::vector<std::pair<KernelId, DimensionVector>>& compositionGlobalSizes,
        const std::vector<std::pair<KernelId, DimensionVector>>& compositionLocalSizes, const std::vector<ParameterPair>& parameterPairs,
        const std::vector<std::pair<KernelId, std::vector<LocalMemoryModifier>>>& compositionLocalMemoryModifiers);

    DimensionVector getGlobalSize() const;
    DimensionVector getLocalSize() const;
    std::vector<LocalMemoryModifier> getLocalMemoryModifiers() const;
    DimensionVector getCompositionKernelGlobalSize(const KernelId id) const;
    DimensionVector getCompositionKernelLocalSize(const KernelId id) const;
    std::vector<LocalMemoryModifier> getCompositionKernelLocalMemoryModifiers(const KernelId id) const;
    std::vector<DimensionVector> getGlobalSizes() const;
    std::vector<DimensionVector> getLocalSizes() const;
    std::vector<ParameterPair> getParameterPairs() const;
    bool isComposite() const;

    friend class PSOSearcher;
    friend std::ostream& operator<<(std::ostream&, const KernelConfiguration&);

private:
    DimensionVector globalSize;
    DimensionVector localSize;
    std::vector<LocalMemoryModifier> localMemoryModifiers;
    std::vector<std::pair<KernelId, DimensionVector>> compositionGlobalSizes;
    std::vector<std::pair<KernelId, DimensionVector>> compositionLocalSizes;
    std::vector<std::pair<KernelId, std::vector<LocalMemoryModifier>>> compositionLocalMemoryModifiers;
    std::vector<ParameterPair> parameterPairs;
    bool compositeConfiguration;
};

std::ostream& operator<<(std::ostream& outputTarget, const KernelConfiguration& configuration);

} // namespace ktt
