#pragma once

#include <string>
#include <vector>

#include "kernel_argument.h"
#include "kernel_parameter.h"
#include "search_method.h"

namespace ktt
{

class Kernel
{
public:
    // Constructor
    explicit Kernel(const std::string& name, const std::string& source);

    // Core methods
    bool addParameter(const KernelParameter& parameter);
    void addArgumentInt(const std::vector<int>& data);
    void addArgumentFloat(const std::vector<float>& data);
    void addArgumentDouble(const std::vector<double>& data);
    bool useSearchMethod(const SearchMethod& searchMethod, const std::vector<double>& searchArguments);

    // Getters
    std::string getName() const;
    std::string getSource() const;
    std::vector<KernelParameter> getParameters() const;
    size_t getArgumentCount() const;
    std::vector<KernelArgument<int>> getArgumentsInt() const;
    std::vector<KernelArgument<float>> getArgumentsFloat() const;
    std::vector<KernelArgument<double>> getArgumentsDouble() const;
    SearchMethod getSearchMethod() const;
    std::vector<double> getSearchArguments() const;

private:
    // Attributes
    std::string name;
    std::string source;
    std::vector<KernelParameter> parameters;
    size_t argumentCount;
    std::vector<KernelArgument<int>> argumentsInt;
    std::vector<KernelArgument<float>> argumentsFloat;
    std::vector<KernelArgument<double>> argumentsDouble;
    SearchMethod searchMethod;
    std::vector<double> searchArguments;

    // Helper methods
    bool parameterExists(const KernelParameter& parameter) const;
};

} // namespace ktt
