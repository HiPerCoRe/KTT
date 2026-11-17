#include <cstdlib>

#include <Api/KttException.h>
#include <Database/Utility/DataDirectory.h>

namespace ktt::db
{

std::filesystem::path GetDefaultDataDirectory()
{
#if defined(_MSC_VER)
// The wide variant is used because the narrow getenv returns the ANSI code page, which cannot represent every user
// name. C4996 (deprecated) is irrelevant here, the value is copied into the path right away.
#pragma warning(suppress : 4996)
    if (const wchar_t *localAppData = _wgetenv(L"LOCALAPPDATA"); localAppData != nullptr && *localAppData != L'\0')
        return std::filesystem::path(localAppData) / "ktt";

    throw KttException("Cannot determine KTT's data directory: LOCALAPPDATA is not set");
#else
    if (const char *dataHome = std::getenv("XDG_DATA_HOME"); dataHome != nullptr && *dataHome != '\0')
        return std::filesystem::path(dataHome) / "ktt";

    if (const char *home = std::getenv("HOME"); home != nullptr && *home != '\0')
        return std::filesystem::path(home) / ".local" / "share" / "ktt";

    throw KttException("Cannot determine KTT's data directory: neither XDG_DATA_HOME nor HOME is set");
#endif // _MSC_VER
}

} // namespace ktt::db
