#pragma once

#include <cstddef>
#include <list>
#include <unordered_map>
#include <utility>

namespace ktt
{

template <typename KeyType, typename ValueType>
class LruCache
{
public:
    using KeyValuePair = typename std::pair<KeyType, ValueType>;
    using ListIterator = typename std::list<KeyValuePair>::iterator;

    LruCache(const size_t maxSize) :
        m_MaxSize(maxSize)
    {}
    
    void Put(const KeyType& key, const ValueType& value)
    {
        PutPrivate(key, value);
    }
    
    void Put(const KeyType& key, ValueType&& value)
    {
        PutPrivate(key, std::move(value));
    }

    ListIterator Get(const KeyType& key)
    {
        auto it = m_ItemsMap.find(key);
        
        if (it == m_ItemsMap.cend())
        {
            return End();
        }

        m_ItemsList.splice(m_ItemsList.begin(), m_ItemsList, it->second);
        return it->second;
    }
    
    void SetMaxSize(const size_t maxSize)
    {
        if (m_MaxSize > maxSize)
        {
            Clear();
        }

        m_MaxSize = maxSize;
    }

    void Clear()
    {
        m_ItemsList.clear();
        m_ItemsMap.clear();
    }

    bool Exists(const KeyType& key) const
    {
        return m_ItemsMap.find(key) != m_ItemsMap.cend();
    }
    
    size_t Size() const noexcept
    {
        return m_ItemsMap.size();
    }

    size_t GetMaxSize() const noexcept
    {
        return m_MaxSize;
    }

    ListIterator Begin()
    {
        return m_ItemsList.begin();
    }
    
    ListIterator End()
    {
        return m_ItemsList.end();
    }

private:
    std::list<KeyValuePair> m_ItemsList;
    std::unordered_map<KeyType, ListIterator> m_ItemsMap;
    size_t m_MaxSize;

    template <typename PrivateValueType>
    void PutPrivate(const KeyType& key, PrivateValueType&& value)
    {
        auto it = m_ItemsMap.find(key);
        m_ItemsList.push_front(KeyValuePair(key, std::forward<PrivateValueType>(value)));

        if (it != m_ItemsMap.cend())
        {
            m_ItemsList.erase(it->second);
            m_ItemsMap.erase(it);
        }

        m_ItemsMap[key] = m_ItemsList.begin();

        if (m_ItemsMap.size() > m_MaxSize)
        {
            auto last = m_ItemsList.end();
            --last;
            m_ItemsMap.erase(last->first);
            m_ItemsList.pop_back();
        }
    }
};

} // namespace ktt
