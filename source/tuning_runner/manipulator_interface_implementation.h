#pragma once

#include <map>

#include "manipulator_interface.h"
#include "../compute_api_driver/compute_api_driver.h"
#include "../dto/kernel_runtime_data.h"
#include "../kernel/kernel_configuration.h"
#include "../kernel_argument/kernel_argument.h"

namespace ktt
{

class ManipulatorInterfaceImplementation : public ManipulatorInterface
{
public:
    // Constructor
    explicit ManipulatorInterfaceImplementation(ComputeApiDriver* computeApiDriver);

    // Inherited methods
    virtual std::vector<ResultArgument> runKernel(const size_t kernelId) override;
    virtual std::vector<ResultArgument> runKernel(const size_t kernelId, const DimensionVector& globalSize,
        const DimensionVector& localSize) override;
    virtual DimensionVector getCurrentGlobalSize(const size_t kernelId) const override;
    virtual DimensionVector getCurrentLocalSize(const size_t kernelId) const override;
    virtual std::vector<ParameterValue> getCurrentConfiguration() const override;
    virtual void updateArgumentScalar(const size_t argumentId, const void* argumentData) override;
    virtual void updateArgumentVector(const size_t argumentId, const void* argumentData) override;
    virtual void updateArgumentVector(const size_t argumentId, const void* argumentData, const size_t numberOfElements) override;
    virtual void setAutomaticArgumentUpdate(const bool flag) override;
    virtual void updateKernelArguments(const size_t kernelId, const std::vector<size_t>& argumentIds) override;
    virtual void swapKernelArguments(const size_t kernelId, const size_t argumentIdFirst, const size_t argumentIdSecond) override;

    // Core methods
    void addKernel(const size_t id, const KernelRuntimeData& kernelRuntimeData);
    void setConfiguration(const KernelConfiguration& kernelConfiguration);
    void setKernelArguments(const std::vector<KernelArgument>& kernelArguments);
    KernelRunResult getCurrentResult() const;
    void clearData();

private:
    // Attributes
    ComputeApiDriver* computeApiDriver;
    KernelRunResult currentResult;
    KernelConfiguration currentConfiguration;
    std::map<size_t, KernelRuntimeData> kernelDataMap;
    std::vector<KernelArgument> kernelArguments;
    bool automaticArgumentUpdate;

    // Helper methods
    void updateArgument(const size_t argumentId, const void* argumentData, const size_t numberOfElements,
        const ArgumentUploadType& argumentUploadType, const bool overrideNumberOfElements);
    std::vector<KernelArgument> getArguments(const std::vector<size_t>& argumentIndices);
};

} // namespace ktt
