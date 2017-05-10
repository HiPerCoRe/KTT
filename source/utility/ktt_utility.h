#pragma once

#include <vector>

#include "../ktt_type_aliases.h"

namespace ktt
{

std::vector<size_t> convertDimensionVector(const DimensionVector& vector);

template <typename T> bool elementExists(const T& element, const std::vector<T>& vector)
{
    for (const auto& currentElement : vector)
    {
        if (currentElement == element)
        {
            return true;
        }
    }
    return false;
}

} // namespace ktt
