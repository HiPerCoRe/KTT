#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <Api/Configuration/ParameterPair.h>
#include <KttTypes.h>

namespace ktt
{

class KernelConstraint
{
public:
    explicit KernelConstraint(const std::vector<std::string>& parameters, ConstraintFunction function);

    const std::vector<std::string>& GetParameters() const;
    bool HasAllParameters(const std::vector<ParameterPair>& pairs) const;
    bool IsFulfilled(const std::vector<ParameterPair>& pairs) const;

private:
    std::vector<std::string> m_Parameters;
    ConstraintFunction m_Function;
};

} // namespace ktt
