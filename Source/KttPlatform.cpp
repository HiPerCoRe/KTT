#include <KttPlatform.h>

namespace ktt
{

static_assert(KTT_VERSION_MAJOR < 1'000, "Invalid major version");
static_assert(KTT_VERSION_MINOR < 1'000, "Invalid minor version");
static_assert(KTT_VERSION_PATCH < 1'000, "Invalid patch version");

uint32_t GetKttVersion()
{
    const uint32_t majorPart = KTT_VERSION_MAJOR * 1'000'000;
    const uint32_t minorPart = KTT_VERSION_MINOR * 1'000;
    const uint32_t patchPart = KTT_VERSION_PATCH;

    return majorPart + minorPart + patchPart;
}

std::string GetKttVersionString()
{
    return std::to_string(KTT_VERSION_MAJOR) + "." + std::to_string(KTT_VERSION_MINOR) + "." + std::to_string(KTT_VERSION_PATCH);
}

} // namespace ktt
