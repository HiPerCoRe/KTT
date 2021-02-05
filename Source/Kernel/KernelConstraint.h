#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <Api/Configuration/ParameterPair.h>

namespace ktt
{

class KernelConstraint
{
public:
    explicit KernelConstraint(const std::vector<std::string>& parameters,
        std::function<bool(const std::vector<uint64_t>&)> function);

    const std::vector<std::string>& GetParameters() const;
    bool HasAllParameters(const std::vector<ParameterPair>& pairs) const;
    bool IsFulfilled(const std::vector<ParameterPair>& pairs) const;

private:
    std::vector<std::string> m_Parameters;
    std::function<bool(const std::vector<uint64_t>&)> m_Function;
};

} // namespace ktt
