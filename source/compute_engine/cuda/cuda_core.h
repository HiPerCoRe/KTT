#pragma once

#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <utility>
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
    explicit CudaCore(const size_t deviceIndex, const size_t queueCount);

    // Kernel execution method
    KernelResult runKernel(const KernelRuntimeData& kernelData, const std::vector<KernelArgument*>& argumentPointers,
        const std::vector<ArgumentOutputDescriptor>& outputDescriptors) override;
    EventId runKernel(const KernelRuntimeData& kernelData, const std::vector<KernelArgument*>& argumentPointers, const QueueId queue) override;
    KernelResult getKernelResult(const EventId id, const std::vector<ArgumentOutputDescriptor>& outputDescriptors) override;

    // Utility methods
    void setCompilerOptions(const std::string& options) override;
    void setGlobalSizeType(const GlobalSizeType& type) override;
    void setAutomaticGlobalSizeCorrection(const bool flag) override;
    void setProgramCache(const bool flag) override;
    void clearProgramCache() override;

    // Queue handling methods
    QueueId getDefaultQueue() const override;
    std::vector<QueueId> getAllQueues() const override;
    void synchronizeQueue(const QueueId queue) override;
    void synchronizeDevice() override;

    // Argument handling methods
    void uploadArgument(KernelArgument& kernelArgument) override;
    void uploadArgument(KernelArgument& kernelArgument, const QueueId queue, const bool synchronizeFlag) override;
    void updateArgument(const ArgumentId id, const void* data, const size_t dataSizeInBytes) override;
    void updateArgument(const ArgumentId id, const void* data, const size_t dataSizeInBytes, const QueueId queue,
        const bool synchronizeFlag) override;
    void downloadArgument(const ArgumentId id, void* destination) const override;
    void downloadArgument(const ArgumentId id, void* destination, const QueueId queue, const bool synchronizeFlag) const override;
    void downloadArgument(const ArgumentId id, void* destination, const size_t dataSizeInBytes) const override;
    void downloadArgument(const ArgumentId id, void* destination, const size_t dataSizeInBytes, const QueueId queue,
        const bool synchronizeFlag) const override;
    KernelArgument downloadArgument(const ArgumentId id) const override;
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
    EventId enqueueKernel(CudaKernel& kernel, const std::vector<size_t>& globalSize, const std::vector<size_t>& localSize,
        const std::vector<CUdeviceptr*>& kernelArguments, const size_t localMemorySize, const QueueId queue);

private:
    size_t deviceIndex;
    size_t queueCount;
    std::string compilerOptions;
    GlobalSizeType globalSizeType;
    bool globalSizeCorrection;
    bool programCacheFlag;
    EventId nextEventId;
    std::unique_ptr<CudaContext> context;
    std::vector<std::unique_ptr<CudaStream>> streams;
    std::set<std::unique_ptr<CudaBuffer>> buffers;
    std::map<std::string, std::unique_ptr<CudaProgram>> programCache;
    std::map<EventId, std::pair<std::unique_ptr<CudaEvent>, std::unique_ptr<CudaEvent>>> kernelEvents;

    DeviceInfo getCudaDeviceInfo(const size_t deviceIndex) const;
    std::vector<CudaDevice> getCudaDevices() const;
    std::vector<CUdeviceptr*> getKernelArguments(const std::vector<KernelArgument*>& argumentPointers);
    size_t getSharedMemorySizeInBytes(const std::vector<KernelArgument*>& argumentPointers) const;
    CudaBuffer* findBuffer(const ArgumentId id) const;
    CUdeviceptr* loadBufferFromCache(const ArgumentId id) const;
};

#else

class CudaCore : public ComputeEngine
{
public:
    // Constructor
    explicit CudaCore(const size_t deviceIndex, const size_t queueCount);

    // Kernel execution method
    KernelResult runKernel(const KernelRuntimeData& kernelData, const std::vector<KernelArgument*>& argumentPointers,
        const std::vector<ArgumentOutputDescriptor>& outputDescriptors) override;
    EventId runKernel(const KernelRuntimeData& kernelData, const std::vector<KernelArgument*>& argumentPointers, const QueueId queue) override;
    KernelResult getKernelResult(const EventId id, const std::vector<ArgumentOutputDescriptor>& outputDescriptors) override;

    // Utility methods
    void setCompilerOptions(const std::string& options) override;
    void setGlobalSizeType(const GlobalSizeType& type) override;
    void setAutomaticGlobalSizeCorrection(const bool flag) override;
    void setProgramCache(const bool flag) override;
    void clearProgramCache() override;

    // Queue handling methods
    QueueId getDefaultQueue() const override;
    std::vector<QueueId> getAllQueues() const override;
    void synchronizeQueue(const QueueId queue) override;
    void synchronizeDevice() override;

    // Argument handling methods
    void uploadArgument(KernelArgument& kernelArgument) override;
    void uploadArgument(KernelArgument& kernelArgument, const QueueId queue, const bool synchronizeFlag) override;
    void updateArgument(const ArgumentId id, const void* data, const size_t dataSizeInBytes) override;
    void updateArgument(const ArgumentId id, const void* data, const size_t dataSizeInBytes, const QueueId queue,
        const bool synchronizeFlag) override;
    void downloadArgument(const ArgumentId id, void* destination) const override;
    void downloadArgument(const ArgumentId id, void* destination, const QueueId queue, const bool synchronizeFlag) const override;
    void downloadArgument(const ArgumentId id, void* destination, const size_t dataSizeInBytes) const override;
    void downloadArgument(const ArgumentId id, void* destination, const size_t dataSizeInBytes, const QueueId queue,
        const bool synchronizeFlag) const override;
    KernelArgument downloadArgument(const ArgumentId id) const override;
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
