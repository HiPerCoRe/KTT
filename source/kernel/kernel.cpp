#include "kernel.h"

namespace ktt
{

Kernel::Kernel(const std::string& name, const std::string& source, const DimensionVector& globalSize, const DimensionVector& localSize):
    name(name),
    source(source),
    globalSize(globalSize),
    localSize(localSize),
    searchMethod(SearchMethod::FullSearch),
    argumentCount(static_cast<size_t>(0))
{}

bool Kernel::addParameter(const KernelParameter& parameter)
{
    if (parameterExists(parameter))
    {
        return false;
    }

    parameters.push_back(parameter);
    return true;
}

void Kernel::addArgumentInt(const std::vector<int>& data)
{
    if (data.size() == 1)
    {
        argumentsInt.push_back(KernelArgument<int>(argumentCount, data, KernelArgumentType::Scalar));
    }
    else
    {
        argumentsInt.push_back(KernelArgument<int>(argumentCount, data, KernelArgumentType::Vector));
    }
    argumentCount++;
}

void Kernel::addArgumentFloat(const std::vector<float>& data)
{
    if (data.size() == 1)
    {
        argumentsFloat.push_back(KernelArgument<float>(argumentCount, data, KernelArgumentType::Scalar));
    }
    else
    {
        argumentsFloat.push_back(KernelArgument<float>(argumentCount, data, KernelArgumentType::Vector));
    }
    argumentCount++;
}

void Kernel::addArgumentDouble(const std::vector<double>& data)
{
    if (data.size() == 1)
    {
        argumentsDouble.push_back(KernelArgument<double>(argumentCount, data, KernelArgumentType::Scalar));
    }
    else
    {
        argumentsDouble.push_back(KernelArgument<double>(argumentCount, data, KernelArgumentType::Vector));
    }
    argumentCount++;
}

bool Kernel::useSearchMethod(const SearchMethod& searchMethod, const std::vector<double>& searchArguments)
{
    if (searchMethod == SearchMethod::RandomSearch && searchArguments.size() < 1
        || searchMethod == SearchMethod::Annealing && searchArguments.size() < 2
        || searchMethod == SearchMethod::PSO && searchArguments.size() < 5)
    {
        return false;
    }
    
    this->searchArguments = searchArguments;
    this->searchMethod = searchMethod;
    return true;
}

std::string Kernel::getName() const
{
    return name;
}

std::string Kernel::getSource() const
{
    return source;
}

DimensionVector Kernel::getGlobalSize() const
{
    return globalSize;
}

DimensionVector Kernel::getLocalSize() const
{
    return localSize;
}

std::vector<KernelParameter> Kernel::getParameters() const
{
    return parameters;
}

size_t Kernel::getArgumentCount() const
{
    return argumentCount;
}

std::vector<KernelArgument<int>> Kernel::getArgumentsInt() const
{
    return argumentsInt;
}

std::vector<KernelArgument<float>> Kernel::getArgumentsFloat() const
{
    return argumentsFloat;
}

std::vector<KernelArgument<double>> Kernel::getArgumentsDouble() const
{
    return argumentsDouble;
}

SearchMethod Kernel::getSearchMethod() const
{
    return searchMethod;
}

std::vector<double> Kernel::getSearchArguments() const
{
    return searchArguments;
}

bool Kernel::parameterExists(const KernelParameter& parameter) const
{
    for (auto& currentParameter : parameters)
    {
        if (currentParameter.getName() == parameter.getName())
        {
            return true;
        }
    }
    return false;
}

} // namespace ktt
