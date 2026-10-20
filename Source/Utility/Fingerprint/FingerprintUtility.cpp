#include <cstdint>

#define XXH_INLINE_ALL
#include <xxhash.h>

#include <Utility/Fingerprint/FingerprintUtility.h>

namespace ktt
{

static_assert(sizeof(std::size_t) == sizeof(XXH64_hash_t), "Fingerprints require 64-bit std::size_t");

std::size_t FingerprintUtility::GetFingerprintOfParameters(const std::set<KernelParameter> &params)
{
    std::size_t base = 0;

    for (const auto &parameter : params)
    {
        base = HashFunction(base, HashString(parameter.GetName()));
    }

    return base;
}

std::size_t FingerprintUtility::GetFingerprintOfDefinitions(const std::vector<const KernelDefinition *> &definitions)
{
    std::size_t base = 0;
    for (const auto *definition : definitions)
    {
        base = HashFunction(base, HashString(definition->GetSource()));
    }
    return base;
}

std::size_t FingerprintUtility::HashString(std::string_view value)
{
    return static_cast<std::size_t>(XXH3_64bits(value.data(), value.size()));
}

std::size_t FingerprintUtility::HashFunction(std::size_t base, std::size_t value)
{
    // Order-sensitive combine; hashing both words through XXH3 gives full avalanche even for small integers
    const std::uint64_t data[2] = {static_cast<std::uint64_t>(base), static_cast<std::uint64_t>(value)};
    return static_cast<std::size_t>(XXH3_64bits(data, sizeof(data)));
}

} // namespace ktt
