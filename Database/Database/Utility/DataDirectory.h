#pragma once

#include <filesystem>

namespace ktt::db
{

/** @fn std::filesystem::path GetDefaultDataDirectory()
 * Returns KTT's per-user data directory: %LOCALAPPDATA%\ktt on Windows, $XDG_DATA_HOME/ktt (falling back to
 * ~/.local/share/ktt) elsewhere. It holds the default local database (ktt.db) and the API key file of the remote
 * database (api-key). The directory is not created.
 * @return Path of the directory.
 * @throw KttException If the location cannot be determined (the environment variables above are unset or empty).
 */
std::filesystem::path GetDefaultDataDirectory();

} // namespace ktt::db
