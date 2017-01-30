#include "kernel.h"

namespace ktt
{

Kernel::Kernel(const std::string& name, const std::string& source):
    name(name),
    source(source),
    searchMethod(SearchMethod::FullSearch),
    argumentsCount(static_cast<size_t>(0))
{}

bool Kernel::addParameter(const std::string& name, const std::vector<size_t>& values)
{
    auto result = parameters.insert( std::pair<std::string, std::vector<size_t>>(name, values));
    return result.second; // return whether insertion was successful
}

void Kernel::addArgumentInt(const std::vector<int>& data)
{
    if (data.size() == 1)
    {
        argumentsInt.push_back(KernelArgument<int>(argumentsCount, data, KernelArgumentType::Scalar));
    }
    else
    {
        argumentsInt.push_back(KernelArgument<int>(argumentsCount, data, KernelArgumentType::Vector));
    }
    argumentsCount++;
}

void Kernel::addArgumentFloat(const std::vector<float>& data)
{
    if (data.size() == 1)
    {
        argumentsFloat.push_back(KernelArgument<float>(argumentsCount, data, KernelArgumentType::Scalar));
    }
    else
    {
        argumentsFloat.push_back(KernelArgument<float>(argumentsCount, data, KernelArgumentType::Vector));
    }
    argumentsCount++;
}

void Kernel::addArgumentDouble(const std::vector<double>& data)
{
    if (data.size() == 1)
    {
        argumentsDouble.push_back(KernelArgument<double>(argumentsCount, data, KernelArgumentType::Scalar));
    }
    else
    {
        argumentsDouble.push_back(KernelArgument<double>(argumentsCount, data, KernelArgumentType::Vector));
    }
    argumentsCount++;
}

std::string Kernel::getName() const
{
    return name;
}

std::string Kernel::getSource() const
{
    return source;
}

SearchMethod Kernel::getSearchMethod() const
{
    return searchMethod;
}

std::vector<double> Kernel::getSearchArguments() const
{
    return searchArguments;
}

std::map<std::string, std::vector<size_t>> Kernel::getParameters() const
{
    return parameters;
}

size_t Kernel::getArgumentsCount() const
{
    return argumentsCount;
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

} // namespace ktt
