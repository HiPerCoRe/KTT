#include <Api/KttException.h>
#include <Kernel/KernelConstraint/BasicConstraint.h>

namespace ktt
{

BasicConstraint::BasicConstraint(const std::vector<const KernelParameter*>& parameters, ConstraintFunction function) :
    KernelConstraint(parameters),
    m_Function(function)
{
    if (m_Function == nullptr)
    {
        throw KttException("Constraint function must be properly defined");
    }

    m_ValuesCache.reserve(m_Parameters.size());
}

bool BasicConstraint::IsFulfilled(const std::vector<const ParameterValue*>& values) const
{
    m_ValuesCache.clear();

    for (const auto& value : values)
    {
        m_ValuesCache.push_back(std::get<uint64_t>(*value));
    }

    return m_Function(m_ValuesCache);
}

} // namespace ktt
