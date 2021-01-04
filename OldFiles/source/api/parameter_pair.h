/** @file parameter_pair.h
  * Functionality related to holding a value for one kernel parameter.
  */
#pragma once

#include <cstddef>
#include <iostream>
#include <string>
#include <ktt_platform.h>

namespace ktt
{

/** @class ParameterPair
  * Class which holds single value for one kernel parameter.
  */
class KTT_API ParameterPair
{
public:
    /** @fn ParameterPair()
      * Default constructor, creates parameter pair with empty name and value set to zero.
      */
    ParameterPair();

    /** @fn explicit ParameterPair(const std::string& name, const size_t value)
      * Constructor which creates parameter pair for integer parameter.
      * @param name Name of a parameter.
      * @param value Value of a parameter.
      */
    explicit ParameterPair(const std::string& name, const size_t value);

    /** @fn explicit ParameterPair(const std::string& name, const double value)
      * Constructor which creates parameter pair for floating-point parameter.
      * @param name Name of a parameter.
      * @param value Value of a parameter.
      */
    explicit ParameterPair(const std::string& name, const double value);

    /** @fn void setValue(const size_t value)
      * Setter for value of an integer parameter.
      * @param value New value of an integer parameter.
      */
    void setValue(const size_t value);

    /** @fn const std::string& getName() const
      * Returns name of a parameter.
      * @return Name of a parameter.
      */
    const std::string& getName() const;

    /** @fn size_t getValue() const
      * Returns integer representation of parameter value.
      * @return Integer representation of parameter value.
      */
    size_t getValue() const;

    /** @fn double getValueDouble() const
      * Returns floating-point representation of parameter value.
      * @return Floating-point representation of parameter value.
      */
    double getValueDouble() const;

    /** @fn bool hasValueDouble() const
      * Checks if parameter value was specified as floating-point.
      * @return True if parameter value was specified as floating-point, false otherwise.
      */
    bool hasValueDouble() const;

private:
    std::string name;
    size_t value;
    double valueDouble;
    bool isDouble;
};

/** @fn std::ostream& operator<<(std::ostream& outputTarget, const ParameterPair& parameterPair)
  * @brief Output operator for parameter pair class.
  * @param outputTarget Location where information about parameter pair will be printed.
  * @param parameterPair Parameter pair object that will be printed.
  * @return Output target to support chaining of output operations.
  */
KTT_API std::ostream& operator<<(std::ostream& outputTarget, const ParameterPair& parameterPair);

} // namespace ktt
