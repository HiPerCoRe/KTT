#include <cmath>
#include <limits>
#include <type_traits>

#include <Utility/NumericalUtilities.h>

namespace ktt
{

template <typename T>
T RoundUp(const T number, const T multiple)
{
    static_assert(std::is_integral_v<T>, "Integral type required.");

    if (multiple <= static_cast<T>(0))
    {
        return number;
    }

    return ((number + multiple - static_cast<T>(1)) / multiple) * multiple;
}

template <typename T>
bool FloatEquals(const T first, const T second, const T epsilon)
{
    return std::fabs(first - second) <= epsilon;
}

template <typename T>
bool FloatEquals(const T first, const T second)
{
    return FloatEquals(first, second, std::numeric_limits<T>::epsilon());
}

} // namespace ktt
