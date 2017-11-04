#pragma once

#include <memory>
#include <string>
#include <vector>
#include "kernel_constraint.h"
#include "kernel_parameter.h"
#include "ktt_types.h"
#include "api/dimension_vector.h"

namespace ktt
{

class Kernel
{
public:
    // Constructor
    explicit Kernel(const KernelId id, const std::string& source, const std::string& name, const DimensionVector& globalSize,
        const DimensionVector& localSize);

    // Core methods
    void addParameter(const KernelParameter& parameter);
    void addConstraint(const KernelConstraint& constraint);
    void setArguments(const std::vector<ArgumentId>& argumentIds);
    void setTuningManipulatorFlag(const bool flag);

    // Getters
    KernelId getId() const;
    std::string getSource() const;
    std::string getName() const;
    DimensionVector getGlobalSize() const;
    DimensionVector getLocalSize() const;
    std::vector<KernelParameter> getParameters() const;
    std::vector<KernelConstraint> getConstraints() const;
    size_t getArgumentCount() const;
    std::vector<ArgumentId> getArgumentIds() const;
    bool hasParameter(const std::string& parameterName) const;
    bool hasTuningManipulator() const;

private:
    // Attributes
    KernelId id;
    std::string source;
    std::string name;
    DimensionVector globalSize;
    DimensionVector localSize;
    std::vector<KernelParameter> parameters;
    std::vector<KernelConstraint> constraints;
    std::vector<ArgumentId> argumentIds;
    bool tuningManipulatorFlag;
};

} // namespace ktt
