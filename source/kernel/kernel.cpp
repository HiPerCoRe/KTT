#include "kernel.h"

namespace ktt
{

Kernel::Kernel(const std::string& source, const std::string& name, const DimensionVector& globalSize, const DimensionVector& localSize):
    source(source),
    name(name),
    globalSize(globalSize),
    localSize(localSize),
    searchMethod(SearchMethod::FullSearch),
    argumentCount(static_cast<size_t>(0))
{}

void Kernel::addParameter(const KernelParameter& parameter)
{
    if (parameterExists(parameter))
    {
        throw std::runtime_error("Parameter with given name already exists: " + parameter.getName());
    }

    parameters.push_back(parameter);
}

void Kernel::addArgumentInt(const std::vector<int>& data)
{
    if (data.size() == 1)
    {
        argumentsInt.push_back(KernelArgument<int>(data, KernelArgumentQuantity::Scalar));
    }
    else
    {
        argumentsInt.push_back(KernelArgument<int>(data, KernelArgumentQuantity::Vector));
    }
    argumentIndices.push_back(ArgumentIndex(argumentCount, KernelArgumentType::Int, argumentsInt.size() - static_cast<size_t>(1)));
    argumentCount++;
}

void Kernel::addArgumentFloat(const std::vector<float>& data)
{
    if (data.size() == 1)
    {
        argumentsFloat.push_back(KernelArgument<float>(data, KernelArgumentQuantity::Scalar));
    }
    else
    {
        argumentsFloat.push_back(KernelArgument<float>(data, KernelArgumentQuantity::Vector));
    }
    argumentIndices.push_back(ArgumentIndex(argumentCount, KernelArgumentType::Float, argumentsFloat.size() - static_cast<size_t>(1)));
    argumentCount++;
}

void Kernel::addArgumentDouble(const std::vector<double>& data)
{
    if (data.size() == 1)
    {
        argumentsDouble.push_back(KernelArgument<double>(data, KernelArgumentQuantity::Scalar));
    }
    else
    {
        argumentsDouble.push_back(KernelArgument<double>(data, KernelArgumentQuantity::Vector));
    }
    argumentIndices.push_back(ArgumentIndex(argumentCount, KernelArgumentType::Double, argumentsDouble.size() - static_cast<size_t>(1)));
    argumentCount++;
}

void Kernel::useSearchMethod(const SearchMethod& searchMethod, const std::vector<double>& searchArguments)
{
    if (searchMethod == SearchMethod::RandomSearch && searchArguments.size() < 1
        || searchMethod == SearchMethod::Annealing && searchArguments.size() < 2
        || searchMethod == SearchMethod::PSO && searchArguments.size() < 5)
    {
        throw std::runtime_error("Insufficient number of arguments given for specified search method: " + getSearchMethodName(searchMethod));
    }
    
    this->searchArguments = searchArguments;
    this->searchMethod = searchMethod;
}

std::string Kernel::getSource() const
{
    return source;
}

std::string Kernel::getName() const
{
    return name;
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

std::vector<ArgumentIndex> Kernel::getArgumentIndices() const
{
    return argumentIndices;
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
    for (const auto& currentParameter : parameters)
    {
        if (currentParameter.getName() == parameter.getName())
        {
            return true;
        }
    }
    return false;
}

std::string Kernel::getSearchMethodName(const SearchMethod& searchMethod) const
{
    switch (searchMethod)
    {
    case SearchMethod::FullSearch:
        return std::string("FullSearch");
    case SearchMethod::RandomSearch:
        return std::string("RandomSearch");
    case SearchMethod::PSO:
        return std::string("PSO");
    case SearchMethod::Annealing:
        return std::string("Annealing");
    default:
        return std::string("Unknown search method");
    }
}

} // namespace ktt
