#pragma once

#include <Kernel/KernelConstraint/KernelConstraint.h>

namespace ktt
{

class GenericConstraint : public KernelConstraint
{
public:
    explicit GenericConstraint(const std::vector<const KernelParameter*>& parameters, GenericConstraintFunction function);

    bool IsFulfilled(const std::vector<const ParameterValue*>& values) const override;

private:
    GenericConstraintFunction m_GenericFunction;
};

} // namespace ktt
