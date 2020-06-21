/** @file user_initializer.h
  * Definition of user initializer class.
  */
#pragma once

#include <vector>
#include <ktt_platform.h>
#include <ktt_types.h>

namespace ktt
{

/** @class UserInitializer
  * Class which can be used to initialize tuner with custom compute device context and queues.
  */
class KTT_API UserInitializer
{
public:
    /** @fn explicit UserInitializer(UserContext context, const std::vector<UserQueue>& queues)
      * Constructor which creates new initializer.
      * @param context User-provided context. Depending on compute API, it can be either CUcontext or cl_context handle.
      * @param queues User-provided queues. Depending on compute API, it can be a vector of either CUstream or cl_command_queue handles.
      * In case of OpenCL API, the queues must be created with CL_QUEUE_PROFILING_ENABLE flag. The number of queues must be at least 1.
      */
    explicit UserInitializer(UserContext context, const std::vector<UserQueue>& queues);

    /** @fn UserContext getContext() const
      * Getter for context. Used internally by KTT tuner.
      * @return User-provided context.
      */
    UserContext getContext() const;

    /** @fn const std::vector<UserQueue>& getQueues() const
      * Getter for queues. Used internally by KTT tuner.
      * @return User-provided queues.
      */
    const std::vector<UserQueue>& getQueues() const;

private:
    UserContext context;
    std::vector<UserQueue> queues;
};

} // namespace ktt
