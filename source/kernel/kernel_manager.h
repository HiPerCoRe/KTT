#pragma once

#include <memory>
#include <vector>

#include "../enums/dimension_vector_type.h"
#include "kernel.h"
#include "kernel_configuration.h"

namespace ktt
{

class KernelManager
{
public:
    // Constructor
    KernelManager();

    // Core methods
    size_t addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize, const DimensionVector& localSize);
    size_t addKernelFromFile(const std::string& filePath, const std::string& kernelName, const DimensionVector& globalSize,
        const DimensionVector& localSize);
    std::string getKernelSourceWithDefines(const size_t id, const KernelConfiguration& kernelConfiguration) const;
    std::vector<KernelConfiguration> getKernelConfigurations(const size_t id) const;

    // Kernel modification methods
    void addParameter(const size_t id, const KernelParameter& parameter);
    void addArgumentInt(const size_t id, const std::vector<int>& data);
    void addArgumentFloat(const size_t id, const std::vector<float>& data);
    void addArgumentDouble(const size_t id, const std::vector<double>& data);
    void useSearchMethod(const size_t id, const SearchMethod& searchMethod, const std::vector<double>& searchArguments);

    // Getters
    size_t getKernelCount() const;
    const std::shared_ptr<const Kernel> getKernel(const size_t id) const;

private:
    // Attributes
    size_t kernelCount;
    std::vector<std::shared_ptr<Kernel>> kernels;

    // Helper methods
    std::string loadFileToString(const std::string& filePath) const;
    void computeConfigurations(const size_t currentParameterIndex, const std::vector<KernelParameter>& parameters,
        const std::vector<ParameterValue>& parameterValues, const DimensionVector& globalSize, const DimensionVector& localSize,
        std::vector<KernelConfiguration>& finalResult) const;
    DimensionVector modifyDimensionVector(const DimensionVector& vector, const DimensionVectorType& dimensionVectorType,
        const KernelParameter& parameter, const size_t parameterValue) const;
};

} // namespace ktt
