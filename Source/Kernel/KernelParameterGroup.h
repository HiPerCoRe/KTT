#pragma once

#include <set>
#include <string>
#include <vector>

#include <Kernel/KernelConstraint.h>
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
    const KernelConstraint& GetNextConstraintToProcess(const std::set<const KernelConstraint*> processedConstraints,
        const std::set<std::string>& processedParameters) const;
    void EnumerateParameterIndices(const std::function<void(std::vector<size_t>&,
        const std::vector<const KernelParameter*>&)>& enumerator) const;

private:
    std::string m_Name;
    std::vector<const KernelParameter*> m_Parameters;
    std::vector<const KernelConstraint*> m_Constraints;

    void ComputeIndices(const size_t currentIndex, std::vector<size_t>& indices,
        const std::vector<const KernelParameter*>& parameters,
        const std::map<size_t, std::vector<const KernelConstraint*>>& evaluationLevels,
        const std::function<void(std::vector<size_t>&, const std::vector<const KernelParameter*>&)>& enumerator) const;
    std::vector<const KernelParameter*> GetParametersInEnumerationOrder() const;
    std::map<size_t, std::vector<const KernelConstraint*>> GetConstraintEvaluationLevels() const;
};

} // namespace ktt
