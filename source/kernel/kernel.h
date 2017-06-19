#pragma once

#include <memory>
#include <string>
#include <vector>

#include "kernel_constraint.h"
#include "kernel_parameter.h"
#include "ktt_type_aliases.h"
#include "enum/search_method.h"

namespace ktt
{

class Kernel
{
public:
    // Constructor
    explicit Kernel(const size_t id, const std::string& source, const std::string& name, const DimensionVector& globalSize,
        const DimensionVector& localSize);

    // Core methods
    void addParameter(const KernelParameter& parameter);
    void addConstraint(const KernelConstraint& constraint);
    void setArguments(const std::vector<size_t>& argumentIndices);
    void setSearchMethod(const SearchMethod& searchMethod, const std::vector<double>& searchArguments);

    // Getters
    size_t getId() const;
    std::string getSource() const;
    std::string getName() const;
    DimensionVector getGlobalSize() const;
    DimensionVector getLocalSize() const;
    std::vector<KernelParameter> getParameters() const;
    std::vector<KernelConstraint> getConstraints() const;
    size_t getArgumentCount() const;
    std::vector<size_t> getArgumentIndices() const;
    SearchMethod getSearchMethod() const;
    std::vector<double> getSearchArguments() const;

private:
    // Attributes
    size_t id;
    std::string source;
    std::string name;
    DimensionVector globalSize;
    DimensionVector localSize;
    std::vector<KernelParameter> parameters;
    std::vector<KernelConstraint> constraints;
    std::vector<size_t> argumentIndices;
    SearchMethod searchMethod;
    std::vector<double> searchArguments;

    // Helper methods
    bool parameterExists(const std::string& parameterName) const;
    std::string getSearchMethodName(const SearchMethod& searchMethod) const;
};

} // namespace ktt
