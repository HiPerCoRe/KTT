/** @file ComputeApiInitializer.h
  * Custom initializer class for compute API context and queues.
  */
#pragma once

#include <vector>

#include <KttPlatform.h>
#include <KttTypes.h>

namespace ktt
{

/** @class ComputeApiInitializer
  * Class which can be used to initialize tuner with custom compute device context and queues.
  */
class KTT_API ComputeApiInitializer
{
public:
    /** @fn explicit ComputeApiInitializer(ComputeContext context, const std::vector<ComputeQueue>& queues)
      * Constructor which creates new initializer.
      * @param context User-provided context. Depending on compute API, it can be either CUcontext or cl_context handle.
      * @param queues User-provided queues. Depending on compute API, it can be a vector of either CUstream or cl_command_queue handles.
      * In case of OpenCL API, the queues must be created with CL_QUEUE_PROFILING_ENABLE flag. The number of queues must be at least 1.
      */
    explicit ComputeApiInitializer(ComputeContext context, const std::vector<ComputeQueue>& queues);

    /** @fn ComputeContext GetContext() const
      * Getter for context. Used internally by KTT framework.
      * @return User-provided context.
      */
    ComputeContext GetContext() const;

    /** @fn const std::vector<ComputeQueue>& GetQueues() const
      * Getter for queues. Used internally by KTT framework.
      * @return User-provided queues.
      */
    const std::vector<ComputeQueue>& GetQueues() const;

private:
    ComputeContext m_Context;
    std::vector<ComputeQueue> m_Queues;
};

} // namespace ktt
