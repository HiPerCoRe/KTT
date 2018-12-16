#pragma once

#include <cstddef>
#include <vector>
#include <api/dimension_vector.h>
#include <api/parameter_pair.h>
#include <ktt_types.h>

namespace ktt
{

class ManipulatorInterface
{
public:
    // Destructor
    virtual ~ManipulatorInterface() = default;

    // Kernel run methods
    virtual void runKernel(const KernelId id) = 0;
    virtual void runKernelAsync(const KernelId id, const QueueId queue) = 0;
    virtual void runKernel(const KernelId id, const DimensionVector& globalSize, const DimensionVector& localSize) = 0;
    virtual void runKernelAsync(const KernelId id, const DimensionVector& globalSize, const DimensionVector& localSize, const QueueId queue) = 0;

    // Kernel profiling methods
    virtual void runKernelWithProfiling(const KernelId id) = 0;
    virtual void runKernelWithProfiling(const KernelId id, const DimensionVector& globalSize, const DimensionVector& localSize) = 0;
    virtual uint64_t getRemainingKernelProfilingRuns(const KernelId id) const = 0;

    // Queue handling methods
    virtual QueueId getDefaultDeviceQueue() const = 0;
    virtual std::vector<QueueId> getAllDeviceQueues() const = 0;
    virtual void synchronizeQueue(const QueueId queue) = 0;
    virtual void synchronizeDevice() = 0;

    // Configuration retrieval methods
    virtual DimensionVector getCurrentGlobalSize(const KernelId id) const = 0;
    virtual DimensionVector getCurrentLocalSize(const KernelId id) const = 0;
    virtual std::vector<ParameterPair> getCurrentConfiguration() const = 0;

    // Argument update and retrieval methods
    virtual void updateArgumentScalar(const ArgumentId id, const void* argumentData) = 0;
    virtual void updateArgumentLocal(const ArgumentId id, const size_t numberOfElements) = 0;
    virtual void updateArgumentVector(const ArgumentId id, const void* argumentData) = 0;
    virtual void updateArgumentVectorAsync(const ArgumentId id, const void* argumentData, const QueueId queue) = 0;
    virtual void updateArgumentVector(const ArgumentId id, const void* argumentData, const size_t numberOfElements) = 0;
    virtual void updateArgumentVectorAsync(const ArgumentId id, const void* argumentData, const size_t numberOfElements, const QueueId queue) = 0;
    virtual void getArgumentVector(const ArgumentId id, void* destination) const = 0;
    virtual void getArgumentVectorAsync(const ArgumentId id, void* destination, const QueueId queue) const = 0;
    virtual void getArgumentVector(const ArgumentId id, void* destination, const size_t numberOfElements) const = 0;
    virtual void getArgumentVectorAsync(const ArgumentId id, void* destination, const size_t numberOfElements, const QueueId queue) const = 0;
    virtual void copyArgumentVector(const ArgumentId destination, const ArgumentId source, const size_t numberOfElements) = 0;
    virtual void copyArgumentVectorAsync(const ArgumentId destination, const ArgumentId source, const size_t numberOfElements,
        const QueueId queue) = 0;
    virtual void resizeArgumentVector(const ArgumentId id, const size_t newNumberOfElements, const bool preserveOldData) = 0;

    // Kernel argument handling methods
    virtual void changeKernelArguments(const KernelId id, const std::vector<ArgumentId>& argumentIds) = 0;
    virtual void swapKernelArguments(const KernelId id, const ArgumentId argumentIdFirst, const ArgumentId argumentIdSecond) = 0;

    // Buffer handling methods
    virtual void createArgumentBuffer(const ArgumentId id) = 0;
    virtual void createArgumentBufferAsync(const ArgumentId id, const QueueId queue) = 0;
    virtual void destroyArgumentBuffer(const ArgumentId id) = 0;
};

} // namespace ktt
