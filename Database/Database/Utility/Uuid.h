#pragma once

#include <array>
#include <cstdint>

namespace ktt::db
{
using uuid = std::array<uint8_t, 16>;

class UuidGenerator
{
public:
    static uuid GenerateUuid();
};


} // namespace ktt::db
