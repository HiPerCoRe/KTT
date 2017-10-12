#pragma once

#include <vector>

#include "kernel.h"

namespace ktt
{

class KernelComposition
{
public:
    // Constructor
    explicit KernelComposition(const size_t id, const std::vector<Kernel*>& kernels);

    // Core methods
    void addParameter(const KernelParameter& parameter);
    void addConstraint(const KernelConstraint& constraint);
    void setArguments(const std::vector<size_t>& argumentIds);

    // Getters
    size_t getId() const;
    std::vector<Kernel*> getKernels() const;

private:
    // Attributes
    size_t id;
    std::vector<Kernel*> kernels;
};

} // namespace ktt
