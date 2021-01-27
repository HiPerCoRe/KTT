#include <Kernel/KernelConstraint.h>

namespace ktt
{

KernelConstraint::KernelConstraint(const std::vector<std::string>& parameters,
    std::function<bool(const std::vector<uint64_t>&)> function) :
    m_Parameters(parameters),
    m_Function(function)
{}

const std::vector<std::string>& KernelConstraint::GetParameters() const
{
    return m_Parameters;
}

bool KernelConstraint::IsFulfilled(const std::vector<ParameterPair>& pairs) const
{
    std::vector<uint64_t> values = ParameterPair::GetParameterValues<uint64_t>(pairs, m_Parameters);
    return m_Function(values);
}

} // namespace ktt
