#pragma once

#include <type_traits>

namespace ktt
{

template <typename EnumType>
struct enable_bitmask_operators
{
    static const bool enable = false;
};

template <typename EnumType>
std::enable_if_t<enable_bitmask_operators<EnumType>::enable, EnumType>
operator&(EnumType lhs, EnumType rhs)
{
    using Underlying = std::underlying_type_t<EnumType>;
    return static_cast<EnumType>(static_cast<Underlying>(lhs) & static_cast<Underlying>(rhs));
}

template <typename EnumType>
std::enable_if_t<enable_bitmask_operators<EnumType>::enable, EnumType>
operator|(EnumType lhs, EnumType rhs)
{
    using Underlying = std::underlying_type_t<EnumType>;
    return static_cast<EnumType>(static_cast<Underlying>(lhs) | static_cast<Underlying>(rhs));
}

template <typename EnumType>
std::enable_if_t<enable_bitmask_operators<EnumType>::enable, EnumType>
operator^(EnumType lhs, EnumType rhs)
{
    using Underlying = std::underlying_type_t<EnumType>;
    return static_cast<EnumType>(static_cast<Underlying>(lhs) ^ static_cast<Underlying>(rhs));
}

template <typename EnumType>
std::enable_if_t<enable_bitmask_operators<EnumType>::enable, EnumType>
operator~(EnumType lhs)
{
    using Underlying = std::underlying_type_t<EnumType>;
    return static_cast<EnumType>(~static_cast<Underlying>(lhs));
}

template <typename EnumType>
std::enable_if_t<enable_bitmask_operators<EnumType>::enable, EnumType&>
operator&=(EnumType& lhs, EnumType rhs)
{
    using Underlying = std::underlying_type_t<EnumType>;
    lhs = static_cast<EnumType>(static_cast<Underlying>(lhs) & static_cast<Underlying>(rhs));
    return lhs;
}

template <typename EnumType>
std::enable_if_t<enable_bitmask_operators<EnumType>::enable, EnumType&>
operator|=(EnumType& lhs, EnumType rhs)
{
    using Underlying = std::underlying_type_t<EnumType>;
    lhs = static_cast<EnumType>(static_cast<Underlying>(lhs) | static_cast<Underlying>(rhs));
    return lhs;
}

template <typename EnumType>
std::enable_if_t<enable_bitmask_operators<EnumType>::enable, EnumType&>
operator^=(EnumType& lhs, EnumType rhs)
{
    using Underlying = std::underlying_type_t<EnumType>;
    lhs = static_cast<EnumType>(static_cast<Underlying>(lhs) ^ static_cast<Underlying>(rhs));
    return lhs;
}

} // namespace ktt
