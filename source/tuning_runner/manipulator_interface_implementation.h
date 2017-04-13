#pragma once

#include "../interface/manipulator_interface.h"
#include "../kernel_argument/argument_manager.h"
#include "../compute_api_driver/opencl/opencl_core.h"

namespace ktt
{

class ManipulatorInterfaceImplementation : public ManipulatorInterface
{
public:
    explicit ManipulatorInterfaceImplementation(ArgumentManager* argumentManager, OpenCLCore* openCLCore);

    virtual std::vector<ResultArgument> runKernel(const size_t kernelId) override;
    virtual std::vector<ResultArgument> runKernel(const size_t kernelId, const DimensionVector& globalSize,
        const DimensionVector& localSize) override;

    virtual void updateArgumentScalar(const size_t argumentId, const void* argumentData) override;
    virtual void updateArgumentVector(const size_t argumentId, const void* argumentData, const size_t dataSizeInBytes) override;

private:
    ArgumentManager* argumentManager;
    OpenCLCore* openCLCore;
};

} // namespace ktt
