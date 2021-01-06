#pragma once

#include <type_traits>

namespace ktt
{

template <typename EnumType>
struct EnableBitfieldOperators
{
    // Bitwise operations are disabled for enums by default.
    static const bool m_Enable = false;
};

template <typename EnumType>
std::enable_if_t<EnableBitfieldOperators<EnumType>::m_Enable, EnumType>
operator&(EnumType lhs, EnumType rhs)
{
    using Underlying = std::underlying_type_t<EnumType>;
    return static_cast<EnumType>(static_cast<Underlying>(lhs) & static_cast<Underlying>(rhs));
}

template <typename EnumType>
std::enable_if_t<EnableBitfieldOperators<EnumType>::m_Enable, EnumType>
operator|(EnumType lhs, EnumType rhs)
{
    using Underlying = std::underlying_type_t<EnumType>;
    return static_cast<EnumType>(static_cast<Underlying>(lhs) | static_cast<Underlying>(rhs));
}

template <typename EnumType>
std::enable_if_t<EnableBitfieldOperators<EnumType>::m_Enable, EnumType>
operator^(EnumType lhs, EnumType rhs)
{
    using Underlying = std::underlying_type_t<EnumType>;
    return static_cast<EnumType>(static_cast<Underlying>(lhs) ^ static_cast<Underlying>(rhs));
}

template <typename EnumType>
std::enable_if_t<EnableBitfieldOperators<EnumType>::m_Enable, EnumType>
operator~(EnumType lhs)
{
    using Underlying = std::underlying_type_t<EnumType>;
    return static_cast<EnumType>(~static_cast<Underlying>(lhs));
}

template <typename EnumType>
std::enable_if_t<EnableBitfieldOperators<EnumType>::m_Enable, EnumType&>
operator&=(EnumType& lhs, EnumType rhs)
{
    using Underlying = std::underlying_type_t<EnumType>;
    lhs = static_cast<EnumType>(static_cast<Underlying>(lhs) & static_cast<Underlying>(rhs));
    return lhs;
}

template <typename EnumType>
std::enable_if_t<EnableBitfieldOperators<EnumType>::m_Enable, EnumType&>
operator|=(EnumType& lhs, EnumType rhs)
{
    using Underlying = std::underlying_type_t<EnumType>;
    lhs = static_cast<EnumType>(static_cast<Underlying>(lhs) | static_cast<Underlying>(rhs));
    return lhs;
}

template <typename EnumType>
std::enable_if_t<EnableBitfieldOperators<EnumType>::m_Enable, EnumType&>
operator^=(EnumType& lhs, EnumType rhs)
{
    using Underlying = std::underlying_type_t<EnumType>;
    lhs = static_cast<EnumType>(static_cast<Underlying>(lhs) ^ static_cast<Underlying>(rhs));
    return lhs;
}

template <typename EnumType>
std::enable_if_t<EnableBitfieldOperators<EnumType>::m_Enable, void>
AddFlag(EnumType& bitfield, EnumType flag)
{
    bitfield |= flag;
}

template <typename EnumType>
std::enable_if_t<EnableBitfieldOperators<EnumType>::m_Enable, void>
RemoveFlag(EnumType& bitfield, EnumType flag)
{
    bitfield &= ~flag;
}

template <typename EnumType>
std::enable_if_t<EnableBitfieldOperators<EnumType>::m_Enable, void>
FlipFlag(EnumType& bitfield, EnumType flag)
{
    bitfield ^= flag;
}

template <typename EnumType>
std::enable_if_t<EnableBitfieldOperators<EnumType>::m_Enable, bool>
HasFlag(EnumType bitfield, EnumType flag)
{
    return static_cast<bool>(bitfield & flag);
}

} // namespace ktt
