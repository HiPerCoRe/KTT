#pragma once

#include <map>
#include <string>
#include <vector>

#include "kernel.h"

namespace ktt
{

class KernelComposition
{
public:
    // Constructor
    explicit KernelComposition(const size_t id, const std::string& name, const std::vector<const Kernel*>& kernels);

    // Core methods
    void addParameter(const KernelParameter& parameter);
    void addConstraint(const KernelConstraint& constraint);
    void setSharedArguments(const std::vector<size_t>& argumentIds);
    void addKernelParameter(const size_t kernelId, const KernelParameter& parameter);
    void setKernelArguments(const size_t kernelId, const std::vector<size_t>& argumentIds);

    // Getters
    size_t getId() const;
    std::string getName() const;
    std::vector<const Kernel*> getKernels() const;
    std::vector<KernelParameter> getParameters() const;
    std::vector<KernelConstraint> getConstraints() const;
    std::vector<size_t> getSharedArgumentIds() const;
    std::vector<size_t> getKernelArgumentIds(const size_t kernelId) const;

private:
    // Attributes
    size_t id;
    std::string name;
    std::vector<const Kernel*> kernels;
    std::vector<KernelParameter> parameters;
    std::vector<KernelConstraint> constraints;
    std::vector<size_t> sharedArgumentIds;
    std::map<size_t, std::vector<size_t>> kernelArgumentIds;

    bool hasParameter(const std::string& parameterName) const;
};

} // namespace ktt
