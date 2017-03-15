#pragma once

#include <string>
#include <vector>

#include "../../libraries/any.hpp"

#include "../ktt_type_aliases.h"
#include "../enums/argument_data_type.h"
#include "../enums/argument_memory_type.h"
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
    template <typename T> void addArgument(const std::vector<T>& data, const ArgumentMemoryType& argumentMemoryType)
    {
        std::vector<linb::any> anyData;
        for (const auto element : data)
        {
            anyData.push_back(element);
        }

        arguments.push_back(KernelArgument(anyData, argumentMemoryType));
    }
    void useSearchMethod(const SearchMethod& searchMethod, const std::vector<double>& searchArguments);

    // Getters
    std::string getSource() const;
    std::string getName() const;
    DimensionVector getGlobalSize() const;
    DimensionVector getLocalSize() const;
    std::vector<KernelParameter> getParameters() const;
    size_t getArgumentCount() const;
    std::vector<KernelArgument> getArguments() const;
    SearchMethod getSearchMethod() const;
    std::vector<double> getSearchArguments() const;

private:
    // Attributes
    std::string source;
    std::string name;
    DimensionVector globalSize;
    DimensionVector localSize;
    std::vector<KernelParameter> parameters;
    std::vector<KernelArgument> arguments;
    SearchMethod searchMethod;
    std::vector<double> searchArguments;

    // Helper methods
    bool parameterExists(const KernelParameter& parameter) const;
    std::string getSearchMethodName(const SearchMethod& searchMethod) const;
};

} // namespace ktt
