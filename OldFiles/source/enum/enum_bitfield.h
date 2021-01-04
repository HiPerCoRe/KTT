/** @file enum_bitfield.h
  * Support for bitwise operations for strongly typed enums.
  */
#pragma once

#include <type_traits>

namespace ktt
{

/** @struct EnableBitfieldOperators
  * Structure which enables bitwise operations support for specified enum.
  */
template <typename EnumType>
struct EnableBitfieldOperators
{
    /** Bitwise operations are disabled by default.
      */
    static const bool enable = false;
};

template <typename EnumType>
std::enable_if_t<EnableBitfieldOperators<EnumType>::enable, EnumType>
operator&(EnumType lhs, EnumType rhs)
{
    using Underlying = std::underlying_type_t<EnumType>;
    return static_cast<EnumType>(static_cast<Underlying>(lhs) & static_cast<Underlying>(rhs));
}

template <typename EnumType>
std::enable_if_t<EnableBitfieldOperators<EnumType>::enable, EnumType>
operator|(EnumType lhs, EnumType rhs)
{
    using Underlying = std::underlying_type_t<EnumType>;
    return static_cast<EnumType>(static_cast<Underlying>(lhs) | static_cast<Underlying>(rhs));
}

template <typename EnumType>
std::enable_if_t<EnableBitfieldOperators<EnumType>::enable, EnumType>
operator^(EnumType lhs, EnumType rhs)
{
    using Underlying = std::underlying_type_t<EnumType>;
    return static_cast<EnumType>(static_cast<Underlying>(lhs) ^ static_cast<Underlying>(rhs));
}

template <typename EnumType>
std::enable_if_t<EnableBitfieldOperators<EnumType>::enable, EnumType>
operator~(EnumType lhs)
{
    using Underlying = std::underlying_type_t<EnumType>;
    return static_cast<EnumType>(~static_cast<Underlying>(lhs));
}

template <typename EnumType>
std::enable_if_t<EnableBitfieldOperators<EnumType>::enable, EnumType&>
operator&=(EnumType& lhs, EnumType rhs)
{
    using Underlying = std::underlying_type_t<EnumType>;
    lhs = static_cast<EnumType>(static_cast<Underlying>(lhs) & static_cast<Underlying>(rhs));
    return lhs;
}

template <typename EnumType>
std::enable_if_t<EnableBitfieldOperators<EnumType>::enable, EnumType&>
operator|=(EnumType& lhs, EnumType rhs)
{
    using Underlying = std::underlying_type_t<EnumType>;
    lhs = static_cast<EnumType>(static_cast<Underlying>(lhs) | static_cast<Underlying>(rhs));
    return lhs;
}

template <typename EnumType>
std::enable_if_t<EnableBitfieldOperators<EnumType>::enable, EnumType&>
operator^=(EnumType& lhs, EnumType rhs)
{
    using Underlying = std::underlying_type_t<EnumType>;
    lhs = static_cast<EnumType>(static_cast<Underlying>(lhs) ^ static_cast<Underlying>(rhs));
    return lhs;
}

template <typename EnumType>
std::enable_if_t<EnableBitfieldOperators<EnumType>::enable, void>
setFlag(EnumType& bitfield, EnumType flag)
{
    bitfield |= flag;
}

template <typename EnumType>
std::enable_if_t<EnableBitfieldOperators<EnumType>::enable, void>
clearFlag(EnumType& bitfield, EnumType flag)
{
    bitfield &= ~flag;
}

template <typename EnumType>
std::enable_if_t<EnableBitfieldOperators<EnumType>::enable, void>
flipFlag(EnumType& bitfield, EnumType flag)
{
    bitfield ^= flag;
}

template <typename EnumType>
std::enable_if_t<EnableBitfieldOperators<EnumType>::enable, bool>
hasFlag(EnumType bitfield, EnumType flag)
{
    return static_cast<bool>(bitfield & flag);
}

} // namespace ktt
