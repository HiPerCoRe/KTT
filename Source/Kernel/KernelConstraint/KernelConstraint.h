#pragma once

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
    KernelConstraint(const std::vector<const KernelParameter*>& parameters);
    virtual ~KernelConstraint() = default;

    const std::vector<const KernelParameter*>& GetParameters() const;
    bool AffectsParameter(const std::string& name) const;
    bool HasAllParameters(const std::set<std::string>& parameterNames) const;
    uint64_t GetAffectedParameterCount(const std::set<std::string>& parameterNames) const;

    virtual bool IsFulfilled(const std::vector<const ParameterValue*>& values) const = 0;

protected:
    std::vector<const KernelParameter*> m_Parameters;
    std::vector<std::string> m_ParameterNames;
};

} // namespace ktt
