#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint> 
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <enum/time_unit.h>

namespace ktt
{

uint64_t convertTime(const uint64_t timeInNanoseconds, const TimeUnit targetUnit);
std::string getTimeUnitTag(const TimeUnit unit);

size_t roundUp(const size_t number, const size_t multiple);
std::vector<size_t> roundUpGlobalSize(const std::vector<size_t>& globalSize, const std::vector<size_t>& localSize);

template <typename T>
bool containsElement(const std::vector<T>& vector, const T& element)
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

template <typename T>
bool containsUnique(const std::vector<T>& vector)
{
    std::set<T> set(vector.begin(), vector.end());
    return set.size() == vector.size();
}

template <typename Key, typename Value>
bool containsKey(const std::map<Key, Value>& map, const Key& key)
{
    return map.find(key) != map.cend();
}

template <typename Key, typename Value>
bool containsValue(const std::map<Key, Value>& map, const Value& value)
{
    for (const auto& pair : map)
    {
        if (pair.second == value)
        {
            return true;
        }
    }

    return false;
}

template <typename Key>
bool containsKey(const std::set<Key>& set, const Key& key)
{
    return set.find(key) != set.cend();
}

template <typename T>
bool floatEquals(const T first, const T second, const T epsilon)
{
    return std::fabs(first - second) <= epsilon;
}

template <typename T>
bool floatEquals(const T first, const T second)
{
    return floatEquals(first, second, std::numeric_limits<T>::epsilon());
}

} // namespace ktt
