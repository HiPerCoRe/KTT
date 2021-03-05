/** @file ComputeInterface.h
  * Functionality related to customizing kernel runs inside KTT.
  */
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <Api/Configuration/DimensionVector.h>
#include <Api/Configuration/KernelConfiguration.h>
#include <KttPlatform.h>
#include <KttTypes.h>

namespace ktt
{

/** @class ComputeInterface
  * Interface for customizing kernel runs in order to run some part of computation on CPU, utilize iterative kernel launches,
  * composite kernels and more. In order to use this functionality, custom kernel launcher function must be defined for the
  * corresponding kernel.
  */
class KTT_API ComputeInterface
{
public:
    /** @fn virtual ~ComputeInterface() = default
      * Compute interface destructor.
      */
    virtual ~ComputeInterface() = default;

    /** @fn virtual void RunKernel(const KernelDefinitionId id) = 0
      * Runs the specified kernel definition using thread sizes based on the current configuration.
      * @param id Id of kernel definition which will be run. The specified definition must be included in the currently launched
      * kernel.
      */
    virtual void RunKernel(const KernelDefinitionId id) = 0;

    /** @fn virtual void RunKernel(const KernelDefinitionId id, const DimensionVector& globalSize, const DimensionVector& localSize) = 0
      * Runs the specified kernel definition using provided thread sizes.
      * @param id Id of kernel definition which will be run. The specified definition must be included in the currently launched
      * kernel.
      * @param globalSize Dimensions for global size with which the kernel will be run.
      * @param localSize Dimensions for local size with which the kernel will be run.
      */
    virtual void RunKernel(const KernelDefinitionId id, const DimensionVector& globalSize, const DimensionVector& localSize) = 0;

    /** @fn virtual ComputeActionId RunKernelAsync(const KernelDefinitionId id, const QueueId queue) = 0
      * Runs the specified kernel definition using thread sizes based on the current configuration. The kernel will be launched
      * asynchronously in the specified queue.
      * @param id Id of kernel definition which will be run. The specified definition must be included in the currently launched
      * kernel.
      * @param queue Id of queue in which the command to run kernel will be submitted.
      * @return Id of asynchronous action corresponding to the issued kernel run command. The action must be waited for with
      * WaitForComputeAction(), SynchronizeQueue() or SynchronizeDevice() methods. Otherwise, problems such as incorrectly recorded
      * kernel durations may occur.
      */
    virtual ComputeActionId RunKernelAsync(const KernelDefinitionId id, const QueueId queue) = 0;

    /** @fn virtual ComputeActionId RunKernelAsync(const KernelDefinitionId id, const QueueId queue, const DimensionVector& globalSize,
      * const DimensionVector& localSize) = 0
      * Runs the specified kernel definition using provided thread sizes. The kernel will be launched asynchronously in the
      * specified queue.
      * @param id Id of kernel definition which will be run. The specified definition must be included in the currently launched
      * kernel.
      * @param queue Id of queue in which the command to run kernel will be submitted.
      * @param globalSize Dimensions for global size with which the kernel will be run.
      * @param localSize Dimensions for local size with which the kernel will be run.
      * @return Id of asynchronous action corresponding to the issued kernel run command. The action must be waited for with
      * WaitForComputeAction(), SynchronizeQueue() or SynchronizeDevice() methods. Otherwise, problems such as incorrectly recorded
      * kernel durations may occur.
      */
    virtual ComputeActionId RunKernelAsync(const KernelDefinitionId id, const QueueId queue, const DimensionVector& globalSize,
        const DimensionVector& localSize) = 0;

    /** @fn virtual void WaitForComputeAction(const ComputeActionId id) = 0
      * Blocks until the specified compute action is finished.
      * @param id Id of compute action to wait for.
      */
    virtual void WaitForComputeAction(const ComputeActionId id) = 0;

    /** @fn virtual void RunKernelWithProfiling(const KernelDefinitionId id) = 0
      * Runs the specified kernel definition using thread sizes based on the current configuration. Collection of kernel profiling
      * counters will be enabled for this run which means that performance will be decreased. Running kernels with profiling will
      * always cause implicit device synchronization before and after the kernel run is finished.
      * @param id Id of kernel definition which will be run. The specified definition must be included in the currently launched
      * kernel.
      */
    virtual void RunKernelWithProfiling(const KernelDefinitionId id) = 0;

    /** @fn virtual void RunKernelWithProfiling(const KernelDefinitionId id, const DimensionVector& globalSize,
      * const DimensionVector& localSize) = 0
      * Runs the specified kernel definition using provided thread sizes. Collection of kernel profiling counters will be enabled
      * for this run which means that performance will be decreased. Running kernels with profiling will always cause implicit
      * device synchronization before and after the kernel run is finished.
      * @param id Id of kernel definition which will be run. The specified definition must be included in the currently launched
      * kernel.
      * @param globalSize Dimensions for global size with which the kernel will be run.
      * @param localSize Dimensions for local size with which the kernel will be run.
      */
    virtual void RunKernelWithProfiling(const KernelDefinitionId id, const DimensionVector& globalSize,
        const DimensionVector& localSize) = 0;

    /** @fn virtual uint64_t GetRemainingProfilingRuns(const KernelDefinitionId id) const = 0
      * Retrieves number of remaining profiling runs that are needed to collect all the profiling counters for the specified
      * kernel definition.
      * @param id Id of kernel definition for which the number of remaining profiling runs will be retrieved. The specified
      * definition must be included in the currently launched kernel.
      * @return Number of remaining profiling runs. Note that if no profiling runs were run so far for the specified definition,
      * zero is returned.
      */
    virtual uint64_t GetRemainingProfilingRuns(const KernelDefinitionId id) const = 0;

    /** @fn virtual uint64_t GetRemainingProfilingRuns() const = 0
      * Retrieves number of remaining profiling runs that are needed to collect all the profiling counters for the currently
      * launched kernel. The number is derived from included kernel definitions which have kernel profiling functionality enabled.
      * @return Number of remaining profiling runs. Note that if no profiling runs were run so far, zero is returned.
      */
    virtual uint64_t GetRemainingProfilingRuns() const = 0;

    /** @fn virtual QueueId GetDefaultQueue() const = 0
      * Retrieves id of default device queue. All synchronous commands are submitted to this queue.
      * @return Id of default device queue.
      */
    virtual QueueId GetDefaultQueue() const = 0;

    /** @fn virtual std::vector<QueueId> GetAllQueues() const = 0
      * Retrieves ids of all available device queues. Number of queues can be specified during tuner creation.
      * @return Ids of all available device queues.
      */
    virtual std::vector<QueueId> GetAllQueues() const = 0;

    /** @fn virtual void SynchronizeQueue(const QueueId queue) = 0
      * Blocks until all commands submitted to the specified device queue are completed.
      * @param queue Id of queue which will be synchronized.
      */
    virtual void SynchronizeQueue(const QueueId queue) = 0;

    /** @fn virtual void SynchronizeDevice() = 0
      * Blocks until all commands submitted to all device queues are completed.
      */
    virtual void SynchronizeDevice() = 0;

    /** @fn virtual const DimensionVector& GetCurrentGlobalSize(const KernelDefinitionId id) const = 0
      * Returns global thread size for the specified kernel definition based on the current configuration.
      * @param id Id of kernel definition for which the global size will be retrieved. The specified definition must be included
      * in the currently launched kernel.
      * @return Global thread size of the specified kernel definition.
      */
    virtual const DimensionVector& GetCurrentGlobalSize(const KernelDefinitionId id) const = 0;

    /** @fn virtual const DimensionVector& GetCurrentLocalSize(const KernelDefinitionId id) const = 0
      * Returns local thread size for the specified kernel definition based on the current configuration.
      * @param id Id of kernel definition for which the local size will be retrieved. The specified definition must be included
      * in the currently launched kernel.
      * @return Local thread size of the specified kernel definition.
      */
    virtual const DimensionVector& GetCurrentLocalSize(const KernelDefinitionId id) const = 0;

    /** @fn virtual const KernelConfiguration& GetCurrentConfiguration() const = 0
      * Returns configuration of the currently launched kernel.
      * @return Configuration of the currently launched kernel. See KernelConfiguration for more information.
      */
    virtual const KernelConfiguration& GetCurrentConfiguration() const = 0;

    /** @fn virtual void ChangeArguments(const KernelDefinitionId id, const std::vector<ArgumentId>& arguments) = 0
      * Changes kernel arguments for the specified kernel definitions under currently launched kernel.
      * @param id Id of kernel definition for which the arguments will be changed. The specified definition must be included
      * in the currently launched kernel.
      * @param arguments Ids of arguments to be used by the specified kernel definition. The order of ids must match the order
      * of arguments inside kernel function. The provided ids must be unique.
      */
    virtual void ChangeArguments(const KernelDefinitionId id, const std::vector<ArgumentId>& arguments) = 0;

    /** @fn virtual void SwapArguments(const KernelDefinitionId id, const ArgumentId first, const ArgumentId second) = 0
      * Swaps positions of kernel arguments for the specified kernel definition under currently launched kernel.
      * @param id Id of kernel definition for which the arguments will be swapped. The specified definition must be included
      * in the currently launched kernel.
      * @param first Id of the first argument which will be swapped.
      * @param second Id of the second argument which will be swapped.
      */
    virtual void SwapArguments(const KernelDefinitionId id, const ArgumentId first, const ArgumentId second) = 0;

    /** @fn virtual void UpdateScalarArgument(const ArgumentId id, const void* data) = 0
      * Updates the specified scalar argument under currently launched kernel.
      * @param id Id of scalar argument which will be updated.
      * @param data Pointer to new data for scalar argument. The data must have matching kernel argument data type.
      */
    virtual void UpdateScalarArgument(const ArgumentId id, const void* data) = 0;

    /** @fn virtual void UpdateLocalArgument(const ArgumentId id, const size_t dataSize) = 0
      * Updates the specified local memory argument under currently launched kernel.
      * @param id Id of local memory argument which will be updated.
      * @param dataSize New size in bytes for the argument.
      */
    virtual void UpdateLocalArgument(const ArgumentId id, const size_t dataSize) = 0;

    /** @fn virtual void UploadBuffer(const ArgumentId id) = 0
      * Uploads the specified vector argument into compute buffer. This method should be used mainly with arguments with
      * ::ArgumentManagementType set to User.
      * @param id Id of vector argument which will be uploaded.
      */
    virtual void UploadBuffer(const ArgumentId id) = 0;

    /** @fn virtual TransferActionId UploadBufferAsync(const ArgumentId id, const QueueId queue) = 0
      * Uploads the specified vector argument into compute buffer. The data will be transferred asynchronously in the specified
      * queue.
      * @param id Id of vector argument which will be uploaded.
      * @param queue Id of queue in which the command to upload argument will be submitted.
      * @return Id of asynchronous action corresponding to the issued data transfer command. The action must be waited for with
      * WaitForTransferAction(), SynchronizeQueue() or SynchronizeDevice() methods. Otherwise, problems such as incorrectly recorded
      * kernel durations may occur.
      */
    virtual TransferActionId UploadBufferAsync(const ArgumentId id, const QueueId queue) = 0;

    /** @fn virtual void DownloadBuffer(const ArgumentId id, void* destination, const size_t dataSize = 0) = 0
      * Downloads the specified vector argument from compute buffer.
      * @param id Id of vector argument which will be downloaded.
      * @param destination Buffer where the argument data will be downloaded. Its size must be equal or greater than the specified
      * data size.
      * @param dataSize Size in bytes of buffer portion which will be downloaded to specified destination, starting with the first
      * byte. If zero, the entire buffer will be downloaded.
      */
    virtual void DownloadBuffer(const ArgumentId id, void* destination, const size_t dataSize = 0) = 0;

    /** @fn virtual TransferActionId DownloadBufferAsync(const ArgumentId id, const QueueId queue, void* destination,
      * const size_t dataSize) = 0
      * Downloads the specified vector argument from compute buffer. The data will be transferred asynchronously in the specified
      * queue.
      * @param id Id of vector argument which will be downloaded.
      * @param queue Id of queue in which the command to download argument will be submitted.
      * @param destination Buffer where the argument data will be downloaded. Its size must be equal or greater than the specified
      * data size.
      * @param dataSize Size in bytes of buffer portion which will be downloaded to specified destination, starting with the first
      * byte. If zero, the entire buffer will be downloaded.
      * @return Id of asynchronous action corresponding to the issued data transfer command. The action must be waited for with
      * WaitForTransferAction(), SynchronizeQueue() or SynchronizeDevice() methods. Otherwise, problems such as incorrectly recorded
      * kernel durations may occur.
      */
    virtual TransferActionId DownloadBufferAsync(const ArgumentId id, const QueueId queue, void* destination,
        const size_t dataSize = 0) = 0;

    /** @fn virtual void UpdateBuffer(const ArgumentId id, const void* data, const size_t dataSize = 0) = 0
      * Updates data in compute buffer of the specified vector argument.
      * @param id Id of vector argument which will be updated.
      * @param data Pointer to new data for vector argument. Its size must be equal or greater than the specified data size.
      * The data must have matching kernel argument data type.
      * @param dataSize Size in bytes of buffer portion which will be updated, starting with the first byte. If zero, the entire
      * buffer will be updated.
      */
    virtual void UpdateBuffer(const ArgumentId id, const void* data, const size_t dataSize = 0) = 0;

    /** @fn virtual TransferActionId UpdateBufferAsync(const ArgumentId id, const QueueId queue, const void* data,
      * const size_t dataSize) = 0
      * Updates data in compute buffer of the specified vector argument. The data will be transferred asynchronously in the
      * specified queue.
      * @param id Id of vector argument which will be updated.
      * @param queue Id of queue in which the command to update argument will be submitted.
      * @param data Pointer to new data for vector argument. Its size must be equal or greater than the specified data size.
      * The data must have matching kernel argument data type.
      * @param dataSize Size in bytes of buffer portion which will be updated, starting with the first byte. If zero, the entire
      * buffer will be updated.
      * @return Id of asynchronous action corresponding to the issued data transfer command. The action must be waited for with
      * WaitForTransferAction(), SynchronizeQueue() or SynchronizeDevice() methods. Otherwise, problems such as incorrectly recorded
      * kernel durations may occur.
      */
    virtual TransferActionId UpdateBufferAsync(const ArgumentId id, const QueueId queue, const void* data,
        const size_t dataSize = 0) = 0;

    /** @fn virtual void CopyBuffer(const ArgumentId destination, const ArgumentId source, const size_t dataSize = 0) = 0
      * Copies part of the compute buffer of source vector argument to compute buffer of destination vector argument.
      * @param destination Id of destination vector argument.
      * @param source Id of source vector argument.
      * @param dataSize Size in bytes of buffer portion which will be copied to destination buffer, starting with the first byte.
      * If zero, the entire buffer will be copied.
      */
    virtual void CopyBuffer(const ArgumentId destination, const ArgumentId source, const size_t dataSize = 0) = 0;

    /** @fn virtual TransferActionId CopyBufferAsync(const ArgumentId destination, const ArgumentId source, const QueueId queue,
      * const size_t dataSize) = 0
      * Copies part of the compute buffer of source vector argument to compute buffer of destination vector argument. The data
      * will be transferred asynchronously in the specified queue.
      * @param destination Id of destination vector argument.
      * @param source Id of source vector argument.
      * @param queue Id of queue in which the command to copy argument will be submitted.
      * @param dataSize Size in bytes of buffer portion which will be copied to destination buffer, starting with the first byte.
      * If zero, the entire buffer will be copied.
      * @return Id of asynchronous action corresponding to the issued data transfer command. The action must be waited for with
      * WaitForTransferAction(), SynchronizeQueue() or SynchronizeDevice() methods. Otherwise, problems such as incorrectly recorded
      * kernel durations may occur.
      */
    virtual TransferActionId CopyBufferAsync(const ArgumentId destination, const ArgumentId source, const QueueId queue,
        const size_t dataSize = 0) = 0;

    /** @fn virtual void WaitForTransferAction(const TransferActionId id) = 0
      * Blocks until the specified buffer transfer action is finished.
      * @param id Id of transfer action to wait for.
      */
    virtual void WaitForTransferAction(const TransferActionId id) = 0;

    /** @fn virtual void ResizeBuffer(const ArgumentId id, const size_t newDataSize, const bool preserveData) = 0
      * Resizes compute buffer for the specified vector argument.
      * @param id Id of vector argument whose buffer will be resized.
      * @param newDataSize Size in bytes for the resized buffer.
      * @param preserveData If true, data from the old buffer will be copied into resized buffer. If false, the old data will be discarded.
      */
    virtual void ResizeBuffer(const ArgumentId id, const size_t newDataSize, const bool preserveData) = 0;

    /** @fn virtual void ClearBuffer(const ArgumentId id) = 0
      * Removes compute buffer for the specified vector argument. This method should be used mainly with arguments with
      * ::ArgumentManagementType set to User.
      * @param id Id of vector argument whose buffer will be removed.
      */
    virtual void ClearBuffer(const ArgumentId id) = 0;

    /** @fn virtual bool HasBuffer(const ArgumentId id) = 0
      * Checks whether compute buffer for the specified vector argument exists.
      * @param id Id of vector argument to check.
      * @return True if the buffer for argument exists. False otherwise.
      */
    virtual bool HasBuffer(const ArgumentId id) = 0;

    /** @fn virtual void GetUnifiedMemoryBufferHandle(const ArgumentId id, UnifiedBufferMemory& memoryHandle) = 0
      * Retrieves memory handle for the specified unified memory argument. The handle can be used to manipulate argument memory
      * on host side. Example usage:
      *     ktt::UnifiedBufferMemory memory;
      *     GetUnifiedMemoryBufferHandle(..., memory);
      *     float* floatArray = static_cast<float*>(memory);
      * @param id Id of vector argument whose memory handle will be retrieved.
      * @param memoryHandle Location where the memory handle will be stored.
      */
    virtual void GetUnifiedMemoryBufferHandle(const ArgumentId id, UnifiedBufferMemory& memoryHandle) = 0;
};

} // namespace ktt
