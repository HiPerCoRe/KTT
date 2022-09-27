#pragma once

#include <Kernel/KernelConstraint/KernelConstraint.h>

namespace ktt
{

class ScriptConstraint : public KernelConstraint
{
public:
    explicit ScriptConstraint(const std::vector<const KernelParameter*>& parameters, const std::string& script);

    bool IsFulfilled(const std::vector<const ParameterValue*>& values) const override;

private:
    std::string m_Script;
};

} // namespace ktt
