#include "kernel.h"

namespace ktt
{

Kernel::Kernel(const std::string& source, const std::string& name, const DimensionVector& globalSize, const DimensionVector& localSize):
    source(source),
    name(name),
    globalSize(globalSize),
    localSize(localSize),
    searchMethod(SearchMethod::FullSearch)
{}

void Kernel::addParameter(const KernelParameter& parameter)
{
    if (parameterExists(parameter))
    {
        throw std::runtime_error("Parameter with given name already exists: " + parameter.getName());
    }

    parameters.push_back(parameter);
}

void Kernel::setArguments(const std::vector<size_t>& argumentIndices)
{
    this->argumentIndices = argumentIndices;
}

void Kernel::setSearchMethod(const SearchMethod& searchMethod, const std::vector<double>& searchArguments)
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
    return argumentIndices.size();
}

std::vector<size_t> Kernel::getArgumentIndices() const
{
    return argumentIndices;
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
