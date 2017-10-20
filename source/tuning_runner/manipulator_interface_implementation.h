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
    void runKernel(const KernelId id) override;
    void runKernel(const KernelId id, const DimensionVector& globalSize, const DimensionVector& localSize) override;
    DimensionVector getCurrentGlobalSize(const KernelId id) const override;
    DimensionVector getCurrentLocalSize(const KernelId id) const override;
    std::vector<ParameterPair> getCurrentConfiguration() const override;
    void updateArgumentScalar(const ArgumentId id, const void* argumentData) override;
    void updateArgumentLocal(const ArgumentId id, const size_t numberOfElements) override;
    void updateArgumentVector(const ArgumentId id, const void* argumentData) override;
    void updateArgumentVector(const ArgumentId id, const void* argumentData, const size_t numberOfElements) override;
    void getArgumentVector(const ArgumentId id, void* destination) const override;
    void getArgumentVector(const ArgumentId id, void* destination, const size_t dataSizeInBytes) const override;
    void changeKernelArguments(const KernelId id, const std::vector<ArgumentId>& argumentIds) override;
    void swapKernelArguments(const KernelId id, const ArgumentId argumentIdFirst, const ArgumentId argumentIdSecond) override;

    // Core methods
    void addKernel(const KernelId id, const KernelRuntimeData& data);
    void setConfiguration(const KernelConfiguration& configuration);
    void setKernelArguments(const std::vector<KernelArgument*>& arguments);
    void uploadBuffers();
    void downloadBuffers(const std::vector<ArgumentOutputDescriptor>& output) const;
    KernelRunResult getCurrentResult() const;
    void clearData();

private:
    // Attributes
    ComputeEngine* computeEngine;
    KernelRunResult currentResult;
    KernelConfiguration currentConfiguration;
    std::map<size_t, KernelRuntimeData> kernelData;
    std::map<size_t, KernelArgument*> vectorArguments;
    std::map<size_t, KernelArgument> nonVectorArguments;

    // Helper methods
    std::vector<KernelArgument*> getArgumentPointers(const std::vector<ArgumentId>& argumentIds);
    void updateArgumentSimple(const ArgumentId id, const void* argumentData, const size_t numberOfElements, const ArgumentUploadType& uploadType);
};

} // namespace ktt
