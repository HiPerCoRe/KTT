#include <Api/KttException.h>
#include <Kernel/KernelConstraint/GenericConstraint.h>

namespace ktt
{

GenericConstraint::GenericConstraint(const std::vector<const KernelParameter*>& parameters, GenericConstraintFunction function) :
    KernelConstraint(parameters),
    m_GenericFunction(function)
{
    if (m_GenericFunction == nullptr)
    {
        throw KttException("Generic constraint function must be properly defined");
    }
}

bool GenericConstraint::IsFulfilled(const std::vector<const ParameterValue*>& values) const
{
    return m_GenericFunction(values);
}

} // namespace ktt
