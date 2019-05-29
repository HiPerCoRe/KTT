/** @file stop_condition.h
  * Interface for implementing tuning stop conditions.
  */
#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <api/computation_result.h>
#include <ktt_types.h>

namespace ktt
{

/** @class StopCondition
  * Class which can be used to stop the tuning process when certain condition is satisfied.
  */
class StopCondition
{
public:
    /** @fn virtual ~StopCondition() = default
      * Stop condition destructor. Inheriting class can override destructor with custom implementation. Default implementation is provided by KTT
      * framework.
      */
    virtual ~StopCondition() = default;

    /** @fn virtual bool isSatisfied() const = 0
      * Checks whether stop condition is already satisfied.
      * @return True if stop condition is satisfied, false otherwise.
      */
    virtual bool isSatisfied() const = 0;

    /** @fn virtual void initialize(const size_t totalConfigurationCount) = 0
      * Performs initialization of stop condition. Called right before the tuning process begins.
      * @param totalConfigurationCount Total count of generated configurations for tuned kernel.
      */
    virtual void initialize(const size_t totalConfigurationCount) = 0;

    /** @fn virtual void updateStatus(const ComputationResult& result) = 0
      * Performs update of stop condition. Called after each tested configuration.
      * @param result Computation result from last tested configuration. See ComputationResult for more information.
      */
    virtual void updateStatus(const ComputationResult& result) = 0;
    
    /** @fn virtual size_t getConfigurationCount() const = 0
      * Returns number of configurations that will be tested before stop condition is satisfied.
      * @return Number of configurations that will be tested before stop condition is satisfied. If unknown, returns total number of generated
      * configurations received in initialization method.
      */
    virtual size_t getConfigurationCount() const = 0;

    /** @fn virtual std::string getStatusString() const = 0
      * Returns stop condition status string. Used for logging.
      * @return Stop condition status string.
      */
    virtual std::string getStatusString() const = 0;
};

} // namespace ktt
