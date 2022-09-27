#pragma once

#include <string>
#include <vector>

#include <Kernel/KernelConstraint/KernelConstraint.h>
#include <Kernel/KernelParameter.h>

namespace ktt
{

class KernelParameterGroup
{
public:
    explicit KernelParameterGroup(const std::string& name, const std::vector<const KernelParameter*>& parameters,
        const std::vector<const KernelConstraint*>& constraints);

    const std::string& GetName() const;
    const std::vector<const KernelParameter*>& GetParameters() const;
    const std::vector<const KernelConstraint*>& GetConstraints() const;
    bool ContainsParameter(const KernelParameter& parameter) const;
    bool ContainsParameter(const std::string& parameter) const;

    std::vector<KernelParameterGroup> GenerateSubgroups() const;
    std::vector<const KernelParameter*> GetParametersInEnumerationOrder() const;
    void EnumerateParameterIndices(const std::function<void(const std::vector<size_t>&)>& enumerator) const;

private:
    std::string m_Name;
    std::vector<const KernelParameter*> m_Parameters;
    std::vector<const KernelConstraint*> m_Constraints;
    mutable std::vector<const ParameterValue*> m_ValuesCache;

    void ComputeIndices(const size_t currentIndex, const std::vector<size_t>& indices,
        const std::map<size_t, std::vector<const KernelConstraint*>>& evaluationLevels,
        const std::vector<const KernelParameter*>& parameters, const std::function<void(const std::vector<size_t>&)>& enumerator) const;
    std::map<size_t, std::vector<const KernelConstraint*>> GetConstraintEvaluationLevels() const;
};

} // namespace ktt
