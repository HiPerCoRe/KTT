#pragma once

#include <cstddef>
#include <map>
#include <set>
#include <vector>

namespace ktt
{

template <typename T, typename Filter>
bool ContainsElementIf(const std::vector<T>& vector, const Filter& filter);

template <typename T>
bool ContainsElement(const std::vector<T>& vector, const T& element);

template <typename T>
bool ContainsUniqueElements(const std::vector<T>& vector);

template <typename Key, typename Value>
bool ContainsKey(const std::map<Key, Value>& map, const Key& key);

template <typename Key, typename Value>
bool ContainsValue(const std::map<Key, Value>& map, const Value& value);

template <typename Key>
bool ContainsKey(const std::set<Key>& set, const Key& key);

template <typename Key, typename Value, typename Filter>
size_t EraseIf(std::map<Key, Value>& map, const Filter& filter);

} // namespace ktt

#include <Utility/StlHelpers.inl>
