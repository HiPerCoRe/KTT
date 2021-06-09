#pragma once

namespace ktt
{

template <typename T>
T RoundUp(const T number, const T multiple);

template <typename T>
bool FloatEquals(const T first, const T second, const T epsilon);

template <typename T>
bool FloatEquals(const T first, const T second);

} // namespace ktt

#include <Utility/NumericalUtilities.inl>
