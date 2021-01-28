#include <Utility/IdGenerator.h>

namespace ktt
{

template <typename IdType>
IdGenerator<IdType>::IdGenerator() :
    m_NextId(static_cast<IdType>(0))
{}

template <typename IdType>
IdGenerator<IdType>::IdGenerator(const IdType initialId) :
    m_NextId(initialId)
{}

template <typename IdType>
IdType IdGenerator<IdType>::GenerateId()
{
    return m_NextId++;
}

} // namespace ktt
