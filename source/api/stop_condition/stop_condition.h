/** @file stop_condition.h
  * ...
  */
#pragma once

#include <cstddef>
#include <string>

namespace ktt
{

/** @class StopCondition
  * ...
  */
class StopCondition
{
public:
    /** @fn virtual ~StopCondition() = default
      * ...
      */
    virtual ~StopCondition() = default;

    /** @fn virtual bool isMet() const = 0
      * ...
      */
    virtual bool isMet() const = 0;

    /** @fn virtual void initialize(const size_t totalConfigurationCount) = 0
      * ...
      */
    virtual void initialize(const size_t totalConfigurationCount) = 0;

    /** @fn virtual void updateStatus(const double previousConfigurationDuration) = 0
      * ...
      */
    virtual void updateStatus(const double previousConfigurationDuration) = 0;
    
    /** @fn virtual std::string getStatusString() const = 0
      * ...
      */
    virtual std::string getStatusString() const = 0;
};

} // namespace ktt
