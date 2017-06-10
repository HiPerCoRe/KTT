#pragma once

#include <set>
#include <vector>

#include "ktt_type_aliases.h"

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

template <typename T> bool containsUnique(const std::vector<T>& vector)
{
    std::set<T> set(vector.begin(), vector.end());
    return set.size() == vector.size();
}

} // namespace ktt
