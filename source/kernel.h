#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "kernel_argument.h"
#include "search_method.h"

namespace ktt
{

class Kernel
{
public:
    explicit Kernel(const std::string& name, const std::string& source);

    bool addParameter(const std::string& name, const std::vector<size_t>& values);
    void addArgumentInt(const std::vector<int>& data);
    void addArgumentFloat(const std::vector<float>& data);
    void addArgumentDouble(const std::vector<double>& data);

    std::string getName() const;
    std::string getSource() const;
    SearchMethod getSearchMethod() const;
    std::vector<double> getSearchArguments() const;
    std::map<std::string, std::vector<size_t>> getParameters() const;

    size_t getArgumentsCount() const;
    std::vector<KernelArgument<int>> getArgumentsInt() const;
    std::vector<KernelArgument<float>> getArgumentsFloat() const;
    std::vector<KernelArgument<double>> getArgumentsDouble() const;

private:
    std::string name;
    std::string source;
    SearchMethod searchMethod;
    std::vector<double> searchArguments;
    std::map<std::string, std::vector<size_t>> parameters;

    size_t argumentsCount;
    std::vector<KernelArgument<int>> argumentsInt;
    std::vector<KernelArgument<float>> argumentsFloat;
    std::vector<KernelArgument<double>> argumentsDouble;
};

} // namespace ktt
