#pragma once

#include "manipulator_interface.h"
#include "../kernel_argument/argument_manager.h"
#include "../compute_api_driver/opencl/opencl_core.h"

namespace ktt
{

class ManipulatorInterfaceImplementation : public ManipulatorInterface
{
public:
    explicit ManipulatorInterfaceImplementation(OpenCLCore* openCLCore);

    virtual std::vector<ResultArgument> runKernel(const size_t kernelId) override;
    virtual std::vector<ResultArgument> runKernel(const size_t kernelId, const DimensionVector& globalSize,
        const DimensionVector& localSize) override;

    virtual void updateArgumentScalar(const size_t argumentId, const void* argumentData) override;
    virtual void updateArgumentVector(const size_t argumentId, const void* argumentData, const size_t dataSizeInBytes) override;

    void setupKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize, const DimensionVector& localSize,
        const std::vector<KernelArgument>& arguments);
    void resetCurrentResult();
    KernelRunResult getCurrentResult() const;

private:
    // Attributes
    OpenCLCore* openCLCore;
    KernelRunResult currentResult;
    std::string kernelSource;
    std::string kernelName;
    DimensionVector globalSize;
    DimensionVector localSize;
    std::vector<KernelArgument> kernelArguments;

    // Helper methods
    std::vector<size_t> convertDimensionVector(const DimensionVector& vector) const;
};

} // namespace ktt
