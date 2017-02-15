#pragma once

#include <string>
#include <vector>

#include "../ktt_type_aliases.h"
#include "../enums/kernel_argument_type.h"
#include "../enums/search_method.h"
#include "kernel_argument.h"
#include "kernel_parameter.h"

namespace ktt
{

class Kernel
{
public:
    // Constructor
    explicit Kernel(const std::string& source, const std::string& name, const DimensionVector& globalSize, const DimensionVector& localSize);

    // Core methods
    void addParameter(const KernelParameter& parameter);
    void addArgumentInt(const std::vector<int>& data);
    void addArgumentFloat(const std::vector<float>& data);
    void addArgumentDouble(const std::vector<double>& data);
    void useSearchMethod(const SearchMethod& searchMethod, const std::vector<double>& searchArguments);

    // Getters
    std::string getSource() const;
    std::string getName() const;
    DimensionVector getGlobalSize() const;
    DimensionVector getLocalSize() const;
    std::vector<KernelParameter> getParameters() const;
    size_t getArgumentCount() const;
    std::vector<ArgumentIndex> getArgumentIndices() const;
    std::vector<KernelArgument<int>> getArgumentsInt() const;
    std::vector<KernelArgument<float>> getArgumentsFloat() const;
    std::vector<KernelArgument<double>> getArgumentsDouble() const;
    SearchMethod getSearchMethod() const;
    std::vector<double> getSearchArguments() const;

private:
    // Attributes
    std::string source;
    std::string name;
    DimensionVector globalSize;
    DimensionVector localSize;
    std::vector<KernelParameter> parameters;
    size_t argumentCount;
    std::vector<ArgumentIndex> argumentIndices;
    std::vector<KernelArgument<int>> argumentsInt;
    std::vector<KernelArgument<float>> argumentsFloat;
    std::vector<KernelArgument<double>> argumentsDouble;
    SearchMethod searchMethod;
    std::vector<double> searchArguments;

    // Helper methods
    bool parameterExists(const KernelParameter& parameter) const;
    std::string getSearchMethodName(const SearchMethod& searchMethod) const;
};

} // namespace ktt
