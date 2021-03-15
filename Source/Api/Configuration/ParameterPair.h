/** @file ParameterPair.h
  * Value for one kernel parameter.
  */
#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include <KttPlatform.h>

namespace ktt
{

/** @class ParameterPair
  * Class which holds single value for one kernel parameter.
  */
class KTT_API ParameterPair
{
public:
    /** @fn ParameterPair()
      * Constructor which creates empty parameter pair.
      */
    ParameterPair();

    /** @fn explicit ParameterPair(const std::string& name, const uint64_t value)
      * Constructor which creates parameter pair for integer parameter.
      * @param name Name of a kernel parameter tied to the pair.
      * @param value Value of a parameter.
      */
    explicit ParameterPair(const std::string& name, const uint64_t value);

    /** @fn explicit ParameterPair(const std::string& name, const double value)
      * Constructor which creates parameter pair for floating-point parameter.
      * @param name Name of a kernel parameter tied to the pair.
      * @param value Value of a parameter.
      */
    explicit ParameterPair(const std::string& name, const double value);

    /** @fn void SetValue(const uint64_t value)
      * Setter for value of an integer parameter.
      * @param value New value of an integer parameter.
      */
    void SetValue(const uint64_t value);

    /** @fn void SetValue(const double value)
      * Setter for value of a floating-point parameter.
      * @param value New value of a floating-point parameter.
      */
    void SetValue(const double value);

    /** @fn const std::string& GetName() const
      * Returns name of a parameter.
      * @return Name of a parameter.
      */
    const std::string& GetName() const;

    /** @fn std::string GetString() const
      * Converts parameter pair to string.
      * @return String in format "parameterName: parameterValue".
      */
    std::string GetString() const;

    /** @fn std::string GetValueString() const
      * Converts parameter value to string.
      * @return Parameter value converted to string.
      */
    std::string GetValueString() const;

    /** @fn uint64_t GetValue() const
      * Returns integer representation of parameter value.
      * @return Integer representation of parameter value.
      */
    uint64_t GetValue() const;

    /** @fn double GetValueDouble() const
      * Returns floating-point representation of parameter value.
      * @return Floating-point representation of parameter value.
      */
    double GetValueDouble() const;

    /** @fn bool HasValueDouble() const
      * Checks if parameter value was specified as floating-point.
      * @return True if parameter value was specified as floating-point, false otherwise.
      */
    bool HasValueDouble() const;

    /** @fn bool HasSameValue(const ParameterPair& other) const
      * Checks if parameter value is same as other parameter value.
      * @param other Source for other value.
      * @return True if the values are same, false otherwise.
      */
    bool HasSameValue(const ParameterPair& other) const;

    /** @fn template <typename T> static T GetParameterValue(const std::vector<ParameterPair>& pairs, const std::string& name)
      * Returns value of the specified parameter from parameter pairs.
      * @param pairs Pairs which will be searched for the specified parameter.
      * @param name Parameter whose value will be returned.
      * @return Value of the specified parameter. Throws KTT exception if the specified parameter was not found.
      */
    template <typename T>
    static T GetParameterValue(const std::vector<ParameterPair>& pairs, const std::string& name);

    /** @fn template <typename T> static std::vector<T> GetParameterValues(const std::vector<ParameterPair>& pairs,
      * const std::vector<std::string>& names)
      * Returns value of all the specified parameters from parameter pairs.
      * @param pairs Pairs which will be searched for the specified parameters.
      * @param names Parameters whose values will be returned.
      * @return Values of the specified parameters. Throws KTT exception if any of the specified parameters was not found.
      */
    template <typename T>
    static std::vector<T> GetParameterValues(const std::vector<ParameterPair>& pairs, const std::vector<std::string>& names);

private:
    std::string m_Name;
    std::variant<uint64_t, double> m_Value;
};

} // namespace ktt

#include <Api/Configuration/ParameterPair.inl>
