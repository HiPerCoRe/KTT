#include <algorithm>
#include <kernel/kernel_constraint.h>

namespace ktt
{

KernelConstraint::KernelConstraint(const std::vector<std::string>& parameterNames,
    const std::function<bool(const std::vector<size_t>&)>& constraintFunction) :
    parameterNames(parameterNames),
    constraintFunction(constraintFunction)
{}

const std::vector<std::string>& KernelConstraint::getParameterNames() const
{
    return parameterNames;
}

bool KernelConstraint::isConfigurationValid(const std::vector<ParameterPair>& configuration) const
{
    std::vector<size_t> constraintValues(parameterNames.size());

    for (size_t i = 0; i < parameterNames.size(); ++i)
    {
        auto iterator = std::find_if(configuration.cbegin(), configuration.cend(), [this, i](const auto& pair)
        {
            return pair.getName() == parameterNames[i];
        });

        if (iterator == configuration.cend())
        {
            // Skip constraint check if some parameter is missing from configuration
            return true;
        }

        constraintValues[i] = iterator->getValue();
    }

    return constraintFunction(constraintValues);
}

} // namespace ktt
