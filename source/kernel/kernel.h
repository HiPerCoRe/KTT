#pragma once

#include <memory>
#include <string>
#include <vector>

#include "../ktt_type_aliases.h"
#include "../enum/search_method.h"
#include "../customization/reference_class.h"
#include "../customization/tuning_manipulator.h"
#include "kernel_constraint.h"
#include "kernel_parameter.h"

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
    void setReferenceKernel(const size_t referenceKernelId, const std::vector<ParameterValue>& referenceKernelConfiguration,
        const std::vector<size_t>& resultArgumentIds);
    void setReferenceClass(std::unique_ptr<ReferenceClass> referenceClass, const std::vector<size_t>& resultArgumentIds);
    void setTuningManipulator(std::unique_ptr<TuningManipulator> tuningManipulator);

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
    bool hasReferenceKernel() const;
    size_t getReferenceKernelId() const;
    std::vector<ParameterValue> getReferenceKernelConfiguration() const;
    std::vector<size_t> getReferenceKernelArgumentIds() const;
    bool hasReferenceClass() const;
    const ReferenceClass* getReferenceClass() const;
    std::vector<size_t> getReferenceClassArgumentIds() const;
    bool hasTuningManipulator() const;
    const TuningManipulator* getTuningManipulator() const;
    TuningManipulator* getTuningManipulator();

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
    bool referenceKernelValid;
    size_t referenceKernelId;
    std::vector<ParameterValue> referenceKernelConfiguration;
    std::vector<size_t> referenceKernelArgumentIds;
    bool referenceClassValid;
    std::unique_ptr<ReferenceClass> referenceClass;
    std::vector<size_t> referenceClassArgumentIds;
    bool tuningManipulatorValid;
    std::unique_ptr<TuningManipulator> tuningManipulator;

    // Helper methods
    bool parameterExists(const std::string& parameterName) const;
    std::string getSearchMethodName(const SearchMethod& searchMethod) const;
};

} // namespace ktt
