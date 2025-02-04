/** @file ParameterPair.h
  * Value for one kernel parameter.
  */
#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include <Kernel/ParameterValueType.h>
#include <KttPlatform.h>
#include <KttTypes.h>

namespace ktt
{

/** @class ParameterPair
  * Class which holds single value for one kernel parameter.
  */
class KTT_API ParameterPair
{
public:
    /** @fn ParameterPair()
      * Constructor which creates an empty parameter pair.
      */
    ParameterPair();

    /** @fn explicit ParameterPair(const std::string& name, const ParameterValue& value)
      * Constructor which creates a parameter pair with the specified value.
      * @param name Name of a kernel parameter tied to the pair.
      * @param value Value of a parameter.
      */
    explicit ParameterPair(const std::string& name, const ParameterValue& value);

    /** @fn void SetValue(const ParameterValue& value)
      * Setter for parameter value.
      * @param value New value of a parameter.
      */
    void SetValue(const ParameterValue& value);

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

    /** @fn const ParameterValue& GetValue() const
      * Returns parameter pair value.
      * @return Parameter pair value.
      */
    const ParameterValue& GetValue() const;

    /** @fn uint64_t GetValueUint() const
      * Returns unsigned integer representation of parameter pair value.
      * @return Unsigned integer representation of parameter pair value.
      */
    uint64_t GetValueUint() const;

    /** @fn ParameterValueType GetValueType() const
      * Returns type of the parameter pair value.
      * @return Type of the parameter pair value.
      */
    ParameterValueType GetValueType() const;

    /** @fn bool HasSameValue(const ParameterPair& other) const
      * Checks if parameter value is same as other parameter value.
      * @param other Source for other value.
      * @return True if the values are same, false otherwise.
      */
    bool HasSameValue(const ParameterPair& other) const;

    /** @fn static ParameterValueType GetTypeFromValue(const ParameterValue& value)
      * Returns type of the specified parameter value.
      * @return Type of the specified parameter value.
      */
    static ParameterValueType GetTypeFromValue(const ParameterValue& value);

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
    ParameterValue m_Value;
};

} // namespace ktt

#include <Api/Configuration/ParameterPair.inl>
