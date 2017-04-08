#pragma once

#include <functional>
#include <string>
#include <vector>

namespace ktt
{

class KernelConstraint
{
public:
    explicit KernelConstraint(const std::function<bool(std::vector<size_t>)>& constraintFunction, const std::vector<std::string>& parameterNames);
    
    std::function<bool(std::vector<size_t>)> getConstraintFunction() const;
    std::vector<std::string> getParameterNames() const;

private:
    std::function<bool(std::vector<size_t>)> constraintFunction;
    std::vector<std::string> parameterNames;
};

} // namespace ktt
