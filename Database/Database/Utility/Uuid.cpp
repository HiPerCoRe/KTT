#include <cstdint>
#include <random>

#include <Database/Utility/Uuid.h>

namespace ktt::db
{

uuid UuidGenerator::GenerateUuid()
{
    std::random_device rd;
    uuid id;

    for (auto& b : id)
        b = static_cast<std::uint8_t>(rd());

    // set version (v4)
    id[6] = (id[6] & 0x0F) | 0x40;

    // set variant
    id[8] = (id[8] & 0x3F) | 0x80;

    return id;
}
} // namespace ktt::db
