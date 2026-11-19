#pragma once

// Internal to the remote database client (not exported): the API key file in KTT's data directory.

#include <filesystem>
#include <string>

namespace ktt::db
{

/** Name of the API key file in KTT's data directory (next to the default local database). */
inline constexpr const char *ApiKeyFileName = "api-key";

/** @fn std::string ReadApiKeyFile(const std::filesystem::path& path)
 * Reads the API key from a key file, without surrounding whitespace.
 * @return The key.
 * @throw KttException If the file does not exist, is not a regular file, cannot be read, is larger than a key can be
 *        or holds only whitespace. The messages name the file, never its content.
 */
std::string ReadApiKeyFile(const std::filesystem::path &path);

} // namespace ktt::db
