#pragma once

#include <cstddef>
#include <set>
#include <vector>

namespace ktt
{

size_t roundUp(const size_t number, const size_t multiple);
std::vector<size_t> roundUpGlobalSize(const std::vector<size_t>& globalSize, const std::vector<size_t>& localSize);

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
