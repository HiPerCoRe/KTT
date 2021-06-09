#pragma once

namespace ktt
{

template <typename IdType>
class IdGenerator
{
public:
    IdGenerator();
    IdGenerator(const IdType initialId);

    IdType GenerateId();

private:
    IdType m_NextId;
};

} // namespace ktt

#include <Utility/IdGenerator.inl>
