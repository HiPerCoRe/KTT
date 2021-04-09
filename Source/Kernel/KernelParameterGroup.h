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

private:
    std::string m_Name;
    std::vector<const KernelParameter*> m_Parameters;
    std::vector<const KernelConstraint*> m_Constraints;
};

} // namespace ktt
