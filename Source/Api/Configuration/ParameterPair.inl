#include <type_traits>

#include <Api/Configuration/ParameterPair.h>
#include <Api/KttException.h>

namespace ktt
{

template <typename T>
T ParameterPair::GetParameterValue(const std::vector<ParameterPair>& pairs, const std::string& name)
{
    static_assert(std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t> || std::is_same_v<T, double>
        || std::is_same_v<T, bool> || std::is_same_v<T, std::string>, "Unsupported kernel parameter value type");

    for (const auto& pair : pairs)
    {
        if (pair.GetName() != name)
        {
            continue;
        }

        if (!std::holds_alternative<T>(pair.m_Value))
        {
            throw KttException("Parameter value type mismatch");
        }

        return std::get<T>(pair.m_Value);
    }

    throw KttException("Parameter with name " + name + " was not found");
}

template <typename T>
std::vector<T> ParameterPair::GetParameterValues(const std::vector<ParameterPair>& pairs, const std::vector<std::string>& names)
{
    std::vector<T> result;

    for (const auto& name : names)
    {
        const T value = GetParameterValue<T>(pairs, name);
        result.push_back(value);
    }

    return result;
}

} // namespace ktt
