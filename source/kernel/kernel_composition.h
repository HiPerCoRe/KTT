#pragma once

#include <vector>

#include "kernel.h"
#include "kernel_constraint.h"
#include "kernel_parameter.h"
#include "ktt_type_aliases.h"

namespace ktt
{

class KernelComposition
{
public:
    // Constructor
    explicit KernelComposition(const size_t id, std::vector<const Kernel*> kernels);

    // Core methods
    void addParameter(const KernelParameter& parameter);
    void addConstraint(const KernelConstraint& constraint);
    void setArguments(const std::vector<size_t>& argumentIds);

    // Getters
    size_t getId() const;
    std::vector<const Kernel*> getKernels() const;
    std::vector<KernelParameter> getParameters() const;
    std::vector<KernelConstraint> getConstraints() const;
    std::vector<KernelConstraint> getConstraintsForKernel(const size_t id) const;
    std::vector<size_t> getArgumentIds() const;
    bool hasCompositeArguments() const;
    bool hasParameter(const std::string& parameterName) const;

private:
    // Attributes
    size_t id;
    std::vector<const Kernel*> kernels;
    std::vector<KernelParameter> parameters;
    std::vector<KernelConstraint> constraints;
    std::vector<std::vector<size_t>> kernelsWithConstraint;
    std::vector<size_t> argumentIds;
    bool compositeArguments;
};

} // namespace ktt
