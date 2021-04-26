#pragma once

#include <cstdint>
#include <functional>
#include <set>
#include <string>
#include <vector>

#include <Api/Configuration/ParameterPair.h>
#include <Kernel/KernelParameter.h>
#include <KttTypes.h>

namespace ktt
{

class KernelConstraint
{
public:
    explicit KernelConstraint(const std::vector<const KernelParameter*>& parameters, ConstraintFunction function);

    const std::vector<const KernelParameter*>& GetParameters() const;
    const std::vector<std::string>& GetParameterNames() const;
    bool AffectsParameter(const std::string& name) const;
    bool HasAllParameters(const std::vector<ParameterPair>& pairs) const;
    bool HasAllParameters(const std::set<std::string>& parameterNames) const;
    uint64_t GetAffectedParameterCount(const std::set<std::string>& parameterNames) const;
    bool IsFulfilled(const std::vector<ParameterPair>& pairs) const;
    bool IsFulfilled(const std::vector<uint64_t>& values) const;

    void EnumeratePairs(const std::function<void(std::vector<ParameterPair>&, const bool)>& enumerator) const;
    void EnumerateParameterIndices(const std::function<void(std::vector<size_t>&, const bool)>& enumerator) const;

private:
    std::vector<const KernelParameter*> m_Parameters;
    std::vector<std::string> m_ParameterNames;
    ConstraintFunction m_Function;

    void ComputePairs(const size_t currentIndex, std::vector<ParameterPair>& pairs,
        const std::function<void(std::vector<ParameterPair>&, const bool)>& enumerator) const;
    void ComputeIndices(const size_t currentIndex, std::vector<size_t>& indices, const std::vector<uint64_t>& values,
        const std::function<void(std::vector<size_t>&, const bool)>& enumerator) const;
};

} // namespace ktt
