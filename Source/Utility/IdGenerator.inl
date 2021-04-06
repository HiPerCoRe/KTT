#include <type_traits>

#include <Utility/IdGenerator.h>

namespace ktt
{

template <typename IdType>
IdGenerator<IdType>::IdGenerator() :
    m_NextId(static_cast<IdType>(0))
{
    static_assert(std::is_integral_v<IdType>, "Id must have integral type.");
}

template <typename IdType>
IdGenerator<IdType>::IdGenerator(const IdType initialId) :
    m_NextId(initialId)
{
    static_assert(std::is_integral_v<IdType>, "Id must have integral type.");
}

template <typename IdType>
IdType IdGenerator<IdType>::GenerateId()
{
    return m_NextId++;
}

} // namespace ktt
