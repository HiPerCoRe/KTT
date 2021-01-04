/** @file kernel_profiling_counter.h
  * Class holding information about single profiling counter.
  */
#pragma once

#include <cstdint>
#include <string>
#include <enum/profiling_counter_type.h>
#include <ktt_platform.h>

namespace ktt
{

/** @union ProfilingCounterValue
  * Union which holds a value of single profiling counter. See ::ProfilingCounterType for more information.
  */
union KTT_API ProfilingCounterValue
{
    /** Corresponds to ProfilingCounterType::Int.
      */
    int64_t intValue;

    /** Corresponds to ProfilingCounterType::UnsignedInt.
      */
    uint64_t uintValue;

    /** Corresponds to ProfilingCounterType::Double.
      */
    double doubleValue;

    /** Corresponds to ProfilingCounterType::Percent.
      */
    double percentValue;

    /** Corresponds to ProfilingCounterType::Throughput.
      */
    uint64_t throughputValue;

    /** Corresponds to ProfilingCounterType::UtilizationLevel.
      */
    uint32_t utilizationLevelValue;
};

/** @class KernelProfilingCounter
  * Class which holds information about single profiling counter.
  */
class KTT_API KernelProfilingCounter
{
public:
    /** @fn explicit KernelProfilingCounter(const std::string& name, const ProfilingCounterValue& value, const ProfilingCounterType type)
      * Constructor which initializes a profiling counter with specified name, value and type.
      * @param name Name of a profiling counter.
      * @param value Value of a profiling counter. See ProfilingCounterValue for more information.
      * @param type Type of a profiling counter. See ::ProfilingCounterType for more information.
      */
    explicit KernelProfilingCounter(const std::string& name, const ProfilingCounterValue& value, const ProfilingCounterType type);

    /** @fn const std::string& getName() const
      * Getter for name of a profiling counter.
      * @return Name of a profiling counter.
      */
    const std::string& getName() const;

    /** @fn const ProfilingCounterValue& getValue() const
      * Getter for value of a profiling counter.
      * @return Value of a profiling counter. See ProfilingCounterValue for more information.
      */
    const ProfilingCounterValue& getValue() const;

    /** @fn ProfilingCounterType getType() const
      * Getter for type of a profiling counter. Type of a profiling counter is used to determine which field inside ProfilingCounterValue needs to
      * accessed in order to retrieve a valid value.
      * @return Type of a profiling counter. See ::ProfilingCounterType for more information.
      */
    ProfilingCounterType getType() const;

private:
    std::string name;
    ProfilingCounterValue value;
    ProfilingCounterType type;
};

} // namespace ktt
