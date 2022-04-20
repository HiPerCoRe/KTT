#pragma once

#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include <Kernel/KernelParameter.h>
#include <KttTypes.h>

namespace ktt
{

class KernelConstraint
{
public:
    explicit KernelConstraint(const std::vector<const KernelParameter*>& parameters, ConstraintFunction function);
    explicit KernelConstraint(const std::vector<const KernelParameter*>& parameters, GenericConstraintFunction function);
    explicit KernelConstraint(const std::vector<const KernelParameter*>& parameters, const std::string& script);

    const std::vector<const KernelParameter*>& GetParameters() const;
    bool AffectsParameter(const std::string& name) const;
    bool HasAllParameters(const std::set<std::string>& parameterNames) const;
    uint64_t GetAffectedParameterCount(const std::set<std::string>& parameterNames) const;
    bool IsFulfilled(const std::vector<const ParameterValue*>& values) const;

private:
    std::vector<const KernelParameter*> m_Parameters;
    std::vector<std::string> m_ParameterNames;
    mutable std::vector<uint64_t> m_ValuesCache;
    ConstraintFunction m_Function;
    GenericConstraintFunction m_GenericFunction;
    std::string m_Script;

    bool EvaluateScript(const std::vector<const ParameterValue*>& values) const;
};

} // namespace ktt
