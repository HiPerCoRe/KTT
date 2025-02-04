#pragma once

#include <Kernel/KernelConstraint/KernelConstraint.h>

namespace ktt
{

class BasicConstraint : public KernelConstraint
{
public:
    explicit BasicConstraint(const std::vector<const KernelParameter*>& parameters, ConstraintFunction function);

    bool IsFulfilled(const std::vector<const ParameterValue*>& values) const override;

private:
    mutable std::vector<uint64_t> m_ValuesCache;
    ConstraintFunction m_Function;
};

} // namespace ktt
