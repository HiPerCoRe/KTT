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
    explicit KernelComposition(const KernelId id, const std::string& name, const std::vector<const Kernel*>& kernels);

    // Core methods
    void addParameter(const KernelParameter& parameter);
    void addConstraint(const KernelConstraint& constraint);
    void setSharedArguments(const std::vector<ArgumentId>& argumentIds);
    void addKernelParameter(const KernelId id, const KernelParameter& parameter);
    void setKernelArguments(const KernelId id, const std::vector<ArgumentId>& argumentIds);
    Kernel transformToKernel() const;

    // Getters
    KernelId getId() const;
    std::string getName() const;
    std::vector<const Kernel*> getKernels() const;
    std::vector<KernelParameter> getParameters() const;
    std::vector<KernelConstraint> getConstraints() const;
    std::vector<ArgumentId> getSharedArgumentIds() const;
    std::vector<ArgumentId> getKernelArgumentIds(const KernelId id) const;
    bool hasParameter(const std::string& parameterName) const;

private:
    // Attributes
    KernelId id;
    std::string name;
    std::vector<const Kernel*> kernels;
    std::vector<KernelParameter> parameters;
    std::vector<KernelConstraint> constraints;
    std::vector<ArgumentId> sharedArgumentIds;
    std::map<KernelId, std::vector<ArgumentId>> kernelArgumentIds;
};

} // namespace ktt
