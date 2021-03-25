/** @file StopCondition.h
  * Interface for implementing tuning stop conditions.
  */
#pragma once

#include <cstdint>
#include <string>

#include <Api/Output/KernelResult.h>
#include <KttPlatform.h>

namespace ktt
{

/** @class StopCondition
  * Class which can be used to stop the tuning process when certain condition is satisfied.
  */
class KTT_API StopCondition
{
public:
    /** @fn virtual ~StopCondition() = default
      * Stop condition destructor. Inheriting class can override destructor with custom implementation. Default implementation
      * is provided by KTT framework.
      */
    virtual ~StopCondition() = default;

    /** @fn virtual bool IsFulfilled() const = 0
      * Checks whether stop condition is fulfilled.
      * @return True if stop condition is fulfilled, false otherwise.
      */
    virtual bool IsFulfilled() const = 0;

    /** @fn virtual void Initialize() = 0
      * Performs initialization of stop condition. Called right before the tuning process begins.
      * @param configurationsCount Total count of generated configurations for the tuned kernel.
      */
    virtual void Initialize(const uint64_t configurationsCount) = 0;

    /** @fn virtual void Update(const KernelResult& result) = 0
      * Performs update of stop condition. Called after each tested configuration.
      * @param result Result from the last tested configuration. See KernelResult for more information.
      */
    virtual void Update(const KernelResult& result) = 0;

    /** @fn virtual std::string GetStatusString() const = 0
      * Returns stop condition status string. Used for logging.
      * @return Stop condition status string.
      */
    virtual std::string GetStatusString() const = 0;
};

} // namespace ktt
