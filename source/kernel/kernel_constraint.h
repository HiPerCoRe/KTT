#pragma once

#include <functional>
#include <string>
#include <vector>
#include <api/parameter_pair.h>

namespace ktt
{

class KernelConstraint
{
public:
    explicit KernelConstraint(const std::vector<std::string>& parameterNames,
        const std::function<bool(const std::vector<size_t>&)>& constraintFunction);

    const std::vector<std::string>& getParameterNames() const;
    bool isConfigurationValid(const std::vector<ParameterPair>& configuration) const;

private:
    std::vector<std::string> parameterNames;
    std::function<bool(const std::vector<size_t>&)> constraintFunction;
};

} // namespace ktt
