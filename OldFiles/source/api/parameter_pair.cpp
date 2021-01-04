#include <api/parameter_pair.h>

namespace ktt
{

ParameterPair::ParameterPair() :
    name(""),
    value(0),
    valueDouble(0.0),
    isDouble(false)
{}

ParameterPair::ParameterPair(const std::string& name, const size_t value) :
    name(name),
    value(value),
    valueDouble(static_cast<double>(value)),
    isDouble(false)
{}

ParameterPair::ParameterPair(const std::string& name, const double value) :
    name(name),
    value(static_cast<size_t>(value)),
    valueDouble(value),
    isDouble(true)
{}

void ParameterPair::setValue(const size_t value)
{
    this->value = value;
    this->valueDouble = static_cast<double>(value);
}

const std::string& ParameterPair::getName() const
{
    return name;
}

size_t ParameterPair::getValue() const
{
    return value;
}

double ParameterPair::getValueDouble() const
{
    return valueDouble;
}

bool ParameterPair::hasValueDouble() const
{
    return isDouble;
}

std::ostream& operator<<(std::ostream& outputTarget, const ParameterPair& parameterPair)
{
    if (!parameterPair.hasValueDouble())
    {
        outputTarget << parameterPair.getName() << ": " << parameterPair.getValue();
    }
    else
    {
        outputTarget << parameterPair.getName() << ": " << parameterPair.getValueDouble();
    }
    return outputTarget;
}

} // namespace ktt
