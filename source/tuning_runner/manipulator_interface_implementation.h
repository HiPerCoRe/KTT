#pragma once

#include <map>
#include <set>
#include <utility>
#include <compute_engine/compute_engine.h>
#include <dto/kernel_runtime_data.h>
#include <kernel/kernel_configuration.h>
#include <kernel_argument/kernel_argument.h>
#include <tuning_runner/manipulator_interface.h>

namespace ktt
{

class ManipulatorInterfaceImplementation : public ManipulatorInterface
{
public:
    // Constructor
    explicit ManipulatorInterfaceImplementation(ComputeEngine* computeEngine);

    // Inherited methods
    void runKernel(const KernelId id) override;
    void runKernelAsync(const KernelId id, const QueueId queue) override;
    void runKernel(const KernelId id, const DimensionVector& globalSize, const DimensionVector& localSize) override;
    void runKernelAsync(const KernelId id, const DimensionVector& globalSize, const DimensionVector& localSize, const QueueId queue) override;
    void runKernelWithProfiling(const KernelId id) override;
    void runKernelWithProfiling(const KernelId id, const DimensionVector& globalSize, const DimensionVector& localSize) override;
    uint64_t getRemainingKernelProfilingRuns(const KernelId id) const override;
    QueueId getDefaultDeviceQueue() const override;
    std::vector<QueueId> getAllDeviceQueues() const override;
    void synchronizeQueue(const QueueId queue) override;
    void synchronizeDevice() override;
    DimensionVector getCurrentGlobalSize(const KernelId id) const override;
    DimensionVector getCurrentLocalSize(const KernelId id) const override;
    std::vector<ParameterPair> getCurrentConfiguration() const override;
    void updateArgumentScalar(const ArgumentId id, const void* argumentData) override;
    void updateArgumentLocal(const ArgumentId id, const size_t numberOfElements) override;
    void updateArgumentVector(const ArgumentId id, const void* argumentData) override;
    void updateArgumentVectorAsync(const ArgumentId id, const void* argumentData, const QueueId queue) override;
    void updateArgumentVector(const ArgumentId id, const void* argumentData, const size_t numberOfElements) override;
    void updateArgumentVectorAsync(const ArgumentId id, const void* argumentData, const size_t numberOfElements, const QueueId queue) override;
    void getArgumentVector(const ArgumentId id, void* destination) const override;
    void getArgumentVectorAsync(const ArgumentId id, void* destination, const QueueId queue) const override;
    void getArgumentVector(const ArgumentId id, void* destination, const size_t numberOfElements) const override;
    void getArgumentVectorAsync(const ArgumentId id, void* destination, const size_t numberOfElements, const QueueId queue) const override;
    void copyArgumentVector(const ArgumentId destination, const ArgumentId source, const size_t numberOfElements) override;
    void copyArgumentVectorAsync(const ArgumentId destination, const ArgumentId source, const size_t numberOfElements, const QueueId queue) override;
    void resizeArgumentVector(const ArgumentId id, const size_t newNumberOfElements, const bool preserveOldData) override;
    void changeKernelArguments(const KernelId id, const std::vector<ArgumentId>& argumentIds) override;
    void swapKernelArguments(const KernelId id, const ArgumentId argumentIdFirst, const ArgumentId argumentIdSecond) override;
    void createArgumentBuffer(const ArgumentId id) override;
    void createArgumentBufferAsync(const ArgumentId id, const QueueId queue) override;
    void destroyArgumentBuffer(const ArgumentId id) override;

    // Core methods
    void setKernelProfiling(const bool flag);
    void addKernel(const KernelId id, const KernelRuntimeData& data);
    void setConfiguration(const KernelConfiguration& configuration);
    void setKernelArguments(const std::vector<KernelArgument*>& arguments);
    void uploadBuffers();
    void downloadBuffers(const std::vector<OutputDescriptor>& output) const;
    KernelResult getCurrentResult() const;
    KernelResult getCurrentResult(const uint64_t remainingProfilingRuns) const;
    void synchronizeDeviceInternal();
    void clearData();
    void resetOverhead();
    void setProfiledKernels(const std::set<KernelId>& profiledKernels);

private:
    // Attributes
    ComputeEngine* computeEngine;
    KernelConfiguration currentConfiguration;
    KernelResult currentResult;
    std::map<KernelId, KernelRuntimeData> kernelData;
    std::map<ArgumentId, KernelArgument*> vectorArguments;
    std::map<ArgumentId, KernelArgument> nonVectorArguments;
    mutable std::map<QueueId, std::set<EventId>> enqueuedKernelEvents;
    mutable std::map<QueueId, std::set<std::pair<EventId, bool>>> enqueuedBufferEvents;
    mutable std::map<KernelId, std::vector<EventId>> kernelProfilingEvents;
    bool kernelProfilingFlag;
    std::set<KernelId> profiledKernels;

    // Helper methods
    std::vector<KernelArgument*> getArgumentPointers(const std::vector<ArgumentId>& argumentIds);
    void updateArgumentSimple(const ArgumentId id, const void* argumentData, const size_t numberOfElements, const ArgumentUploadType uploadType);
    void storeKernelEvent(const QueueId queue, const EventId event) const;
    void storeKernelProfilingEvent(const KernelId kernel, const EventId event) const;
    void storeBufferEvent(const QueueId queue, const EventId event, const bool increaseOverhead) const;
    void processKernelEvents(const std::set<EventId>& events);
    void processBufferEvents(const std::set<std::pair<EventId, bool>>& events);
};

} // namespace ktt
