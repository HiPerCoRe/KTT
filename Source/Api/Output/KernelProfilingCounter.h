/** @file KernelProfilingCounter.h
  * Information about a single profiling counter.
  */
#pragma once

#include <cstdint>
#include <string>
#include <variant>

#include <Api/Output/ProfilingCounterType.h>
#include <KttPlatform.h>

namespace ktt
{

/** @class KernelProfilingCounter
  * Class which holds information about single profiling counter.
  */
class KTT_API KernelProfilingCounter
{
public:
    /** @fn KernelProfilingCounter() = default
      * Constructor which creates empty profiling counter.
      */
    KernelProfilingCounter() = default;

    /** @fn explicit KernelProfilingCounter(const std::string& name, const ProfilingCounterType type, const int64_t value)
      * Constructor which initializes a profiling counter with specified name, value and type.
      * @param name Name of a profiling counter.
      * @param type Type of a profiling counter. See ::ProfilingCounterType for more information.
      * @param value Integer value of a profiling counter.
      */
    explicit KernelProfilingCounter(const std::string& name, const ProfilingCounterType type, const int64_t value);

    /** @fn explicit KernelProfilingCounter(const std::string& name, const ProfilingCounterType type, const uint64_t value)
      * Constructor which initializes a profiling counter with specified name, value and type.
      * @param name Name of a profiling counter.
      * @param type Type of a profiling counter. See ::ProfilingCounterType for more information.
      * @param value Unsigned integer value of a profiling counter.
      */
    explicit KernelProfilingCounter(const std::string& name, const ProfilingCounterType type, const uint64_t value);

    /** @fn explicit KernelProfilingCounter(const std::string& name, const ProfilingCounterType type, const double value)
      * Constructor which initializes a profiling counter with specified name, value and type.
      * @param name Name of a profiling counter.
      * @param type Type of a profiling counter. See ::ProfilingCounterType for more information.
      * @param value Double value of a profiling counter.
      */
    explicit KernelProfilingCounter(const std::string& name, const ProfilingCounterType type, const double value);

    /** @fn const std::string& GetName() const
      * Getter for name of a profiling counter.
      * @return Name of a profiling counter.
      */
    const std::string& GetName() const;

    /** @fn ProfilingCounterType GetType() const
      * Getter for type of a profiling counter. Type of a profiling counter is used to determine which field inside
      * ProfilingCounterValue needs to accessed in order to retrieve a valid value.
      * @return Type of a profiling counter. See ::ProfilingCounterType for more information.
      */
    ProfilingCounterType GetType() const;

    /** @fn int64_t GetValueInt() const
      * Getter for integer value of a profiling counter. Should be used if counter type is Int.
      * @return Integer value of a profiling counter.
      */
    int64_t GetValueInt() const;

    /** @fn uint64_t GetValueUint() const
      * Getter for unsigned  integer value of a profiling counter. Should be used if counter type is UnsignedInt, Throughput
      * or UtilizationLevel.
      * @return Unsigned integer value of a profiling counter.
      */
    uint64_t GetValueUint() const;

    /** @fn double GetValueDouble() const
      * Getter for double value of a profiling counter. Should be used if counter type is Double or Percent.
      * @return Double value of a profiling counter.
      */
    double GetValueDouble() const;

    /** @fn bool operator==(const KernelProfilingCounter& other) const
      * Checks whether profiling counter is equal to other. I.e., counters have the same name.
      * @param other Comparison target.
      * @return True if the counters are equal. False otherwise.
      */
    bool operator==(const KernelProfilingCounter& other) const;

    /** @fn bool operator!=(const KernelProfilingCounter& other) const
      * Checks whether profiling counter is not equal to other. I.e., counters do not have the same name.
      * @param other Comparison target.
      * @return True if the counters are not equal. False otherwise.
      */
    bool operator!=(const KernelProfilingCounter& other) const;

    /** @fn bool operator<(const KernelProfilingCounter& other) const
      * Checks whether profiling counter is lesser than other. I.e., its name is lesser than other's name.
      * @param other Comparison target.
      * @return True if the counter is lesser than other. False otherwise.
      */
    bool operator<(const KernelProfilingCounter& other) const;

private:
    std::string m_Name;
    ProfilingCounterType m_Type;
    std::variant<int64_t, uint64_t, double> m_Value;
};

} // namespace ktt
