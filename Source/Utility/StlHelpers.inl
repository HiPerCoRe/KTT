#include <map>
#include <set>
#include <vector>

#include <Utility/StlHelpers.h>

namespace ktt
{

template <typename T, typename Filter>
bool ContainsElementIf(const std::vector<T>& vector, const Filter& filter)
{
    for (const auto& currentElement : vector)
    {
        if (filter(currentElement))
        {
            return true;
        }
    }

    return false;
}

template <typename T>
bool ContainsElement(const std::vector<T>& vector, const T& element)
{
    return ContainsElementIf(vector, [&element](const auto& currentElement)
    {
        return currentElement == element;
    });
}

template <typename T>
bool ContainsUniqueElements(const std::vector<T>& vector)
{
    std::set<T> set(vector.begin(), vector.end());
    return set.size() == vector.size();
}

template <typename Key, typename Value>
bool ContainsKey(const std::map<Key, Value>& map, const Key& key)
{
    return map.find(key) != map.cend();
}

template <typename Key, typename Value>
bool ContainsValue(const std::map<Key, Value>& map, const Value& value)
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
bool ContainsKey(const std::set<Key>& set, const Key& key)
{
    return set.find(key) != set.cend();
}

} // namespace ktt
