#pragma once

#include <map>

#include "manipulator_interface.h"
#include "compute_engine/compute_engine.h"
#include "dto/kernel_runtime_data.h"
#include "kernel/kernel_configuration.h"
#include "kernel_argument/kernel_argument.h"

namespace ktt
{

class ManipulatorInterfaceImplementation : public ManipulatorInterface
{
public:
    // Constructor
    explicit ManipulatorInterfaceImplementation(ComputeEngine* computeEngine);

    // Inherited methods
    void runKernel(const size_t kernelId) override;
    void runKernel(const size_t kernelId, const DimensionVector& globalSize, const DimensionVector& localSize) override;
    DimensionVector getCurrentGlobalSize(const size_t kernelId) const override;
    DimensionVector getCurrentLocalSize(const size_t kernelId) const override;
    std::vector<ParameterValue> getCurrentConfiguration() const override;
    void updateArgumentScalar(const size_t argumentId, const void* argumentData) override;
    void updateArgumentLocal(const size_t argumentId, const size_t numberOfElements) override;
    void updateArgumentVector(const size_t argumentId, const void* argumentData) override;
    void updateArgumentVector(const size_t argumentId, const void* argumentData, const size_t numberOfElements) override;
    void getArgumentVector(const size_t argumentId, void* destination) const override;
    void getArgumentVector(const size_t argumentId, void* destination, const size_t dataSizeInBytes) const override;
    void changeKernelArguments(const size_t kernelId, const std::vector<size_t>& argumentIds) override;
    void swapKernelArguments(const size_t kernelId, const size_t argumentIdFirst, const size_t argumentIdSecond) override;

    // Core methods
    void addKernel(const size_t id, const KernelRuntimeData& kernelRuntimeData);
    void setConfiguration(const KernelConfiguration& kernelConfiguration);
    void setKernelArguments(const std::vector<KernelArgument*>& kernelArguments);
    void uploadBuffers();
    void downloadBuffers(const std::vector<ArgumentOutputDescriptor>& outputDescriptors) const;
    KernelRunResult getCurrentResult() const;
    void clearData();

private:
    // Attributes
    ComputeEngine* computeEngine;
    KernelRunResult currentResult;
    KernelConfiguration currentConfiguration;
    std::map<size_t, KernelRuntimeData> kernelDataMap;
    std::map<size_t, KernelArgument*> vectorArgumentMap;
    std::map<size_t, KernelArgument> nonVectorArgumentMap;

    // Helper methods
    std::vector<KernelArgument*> getArgumentPointers(const std::vector<size_t>& argumentIndices);
    void updateArgumentSimple(const size_t argumentId, const void* argumentData, const size_t numberOfElements,
        const ArgumentUploadType& argumentUploadType);
};

} // namespace ktt
