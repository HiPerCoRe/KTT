#pragma once

#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <vector>
#include "compute_engine/compute_engine.h"

#ifdef PLATFORM_CUDA
#include "cuda.h"
#include "nvrtc.h"
#include "cuda_buffer.h"
#include "cuda_context.h"
#include "cuda_device.h"
#include "cuda_event.h"
#include "cuda_kernel.h"
#include "cuda_program.h"
#include "cuda_stream.h"
#include "cuda_utility.h"
#endif // PLATFORM_CUDA

namespace ktt
{

#ifdef PLATFORM_CUDA

class CudaCore : public ComputeEngine
{
public:
    // Constructor
    explicit CudaCore(const size_t deviceIndex);

    // Kernel execution method
    KernelResult runKernel(const QueueId queue, const KernelRuntimeData& kernelData, const std::vector<KernelArgument*>& argumentPointers,
        const std::vector<ArgumentOutputDescriptor>& outputDescriptors) override;

    // Utility methods
    void setCompilerOptions(const std::string& options) override;
    void setGlobalSizeType(const GlobalSizeType& type) override;
    void setAutomaticGlobalSizeCorrection(const bool flag) override;

    // Queue handling methods
    QueueId getDefaultQueue() const override;
    QueueId createQueue() override;

    // Argument handling methods
    void uploadArgument(const QueueId queue, KernelArgument& kernelArgument) override;
    void updateArgument(const QueueId queue, const ArgumentId id, const void* data, const size_t dataSizeInBytes) override;
    KernelArgument downloadArgument(const QueueId queue, const ArgumentId id) const override;
    void downloadArgument(const QueueId queue, const ArgumentId id, void* destination) const override;
    void downloadArgument(const QueueId queue, const ArgumentId id, void* destination, const size_t dataSizeInBytes) const override;
    void clearBuffer(const ArgumentId id) override;
    void clearBuffers() override;
    void clearBuffers(const ArgumentAccessType& accessType) override;

    // Information retrieval methods
    void printComputeApiInfo(std::ostream& outputTarget) const override;
    std::vector<PlatformInfo> getPlatformInfo() const override;
    std::vector<DeviceInfo> getDeviceInfo(const size_t platformIndex) const override;
    DeviceInfo getCurrentDeviceInfo() const override;

    // Low-level kernel execution methods
    std::unique_ptr<CudaProgram> createAndBuildProgram(const std::string& source) const;
    std::unique_ptr<CudaEvent> createEvent() const;
    std::unique_ptr<CudaKernel> createKernel(const CudaProgram& program, const std::string& kernelName) const;
    float enqueueKernel(const QueueId queue, CudaKernel& kernel, const std::vector<size_t>& globalSize, const std::vector<size_t>& localSize,
        const std::vector<CUdeviceptr*>& kernelArguments, const size_t localMemorySize) const;

private:
    size_t deviceIndex;
    std::string compilerOptions;
    GlobalSizeType globalSizeType;
    bool globalSizeCorrection;
    QueueId nextId;
    std::unique_ptr<CudaContext> context;
    std::vector<std::unique_ptr<CudaStream>> streams;
    std::set<std::unique_ptr<CudaBuffer>> buffers;

    DeviceInfo getCudaDeviceInfo(const size_t deviceIndex) const;
    std::vector<CudaDevice> getCudaDevices() const;
    std::vector<CUdeviceptr*> getKernelArguments(const QueueId queue, const std::vector<KernelArgument*>& argumentPointers);
    size_t getSharedMemorySizeInBytes(const std::vector<KernelArgument*>& argumentPointers) const;
    CudaBuffer* findBuffer(const ArgumentId id) const;
    CUdeviceptr* loadBufferFromCache(const ArgumentId id) const;
};

#else

class CudaCore : public ComputeEngine
{
public:
    // Constructor
    explicit CudaCore(const size_t deviceIndex);

    // Kernel execution method
    KernelResult runKernel(const QueueId queue, const KernelRuntimeData& kernelData, const std::vector<KernelArgument*>& argumentPointers,
        const std::vector<ArgumentOutputDescriptor>& outputDescriptors) override;

    // Utility methods
    void setCompilerOptions(const std::string& options) override;
    void setGlobalSizeType(const GlobalSizeType& type) override;
    void setAutomaticGlobalSizeCorrection(const bool flag) override;

    // Queue handling methods
    QueueId getDefaultQueue() const override;
    QueueId createQueue() override;

    // Argument handling methods
    void uploadArgument(const QueueId queue, KernelArgument& kernelArgument) override;
    void updateArgument(const QueueId queue, const ArgumentId id, const void* data, const size_t dataSizeInBytes) override;
    KernelArgument downloadArgument(const QueueId queue, const ArgumentId id) const override;
    void downloadArgument(const QueueId queue, const ArgumentId id, void* destination) const override;
    void downloadArgument(const QueueId queue, const ArgumentId id, void* destination, const size_t dataSizeInBytes) const override;
    void clearBuffer(const ArgumentId id) override;
    void clearBuffers() override;
    void clearBuffers(const ArgumentAccessType& accessType) override;

    // Information retrieval methods
    void printComputeApiInfo(std::ostream& outputTarget) const override;
    std::vector<PlatformInfo> getPlatformInfo() const override;
    std::vector<DeviceInfo> getDeviceInfo(const size_t platformIndex) const override;
    DeviceInfo getCurrentDeviceInfo() const override;
};

#endif // PLATFORM_CUDA

} // namespace ktt
