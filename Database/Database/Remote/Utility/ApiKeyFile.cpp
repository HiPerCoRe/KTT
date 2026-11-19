#include <cstddef>
#include <cstdint>
#include <fstream>
#include <system_error>

#include <Api/KttException.h>
#include <Database/Remote/Utility/ApiKeyFile.h>

namespace ktt::db
{

namespace
{

// A key is a short token; anything larger is not a key file.
constexpr std::uintmax_t MaxApiKeyFileSize = 4096;

} // namespace

std::string ReadApiKeyFile(const std::filesystem::path &path)
{
    const std::string where = "RemoteDatabase: the API key file " + path.u8string();
    std::error_code error;

    if (!std::filesystem::exists(path, error))
    {
        throw KttException(where + " does not exist; write the key to it or pass the key to the RemoteDatabase "
            "constructor");
    }

    if (!std::filesystem::is_regular_file(path, error))
        throw KttException(where + " is not a file");

    const std::uintmax_t size = std::filesystem::file_size(path, error);

    if (error || size > MaxApiKeyFileSize)
        throw KttException(where + (error ? " cannot be read" : " is too large to hold an API key"));

    std::ifstream file(path, std::ios::binary);
    std::string text(static_cast<size_t>(size), '\0');

    if (!file || !file.read(text.data(), static_cast<std::streamsize>(text.size())))
        throw KttException(where + " cannot be read");

    static constexpr const char *Whitespace = " \t\r\n";
    const size_t first = text.find_first_not_of(Whitespace);

    if (first == std::string::npos)
        throw KttException(where + " is empty");

    const size_t last = text.find_last_not_of(Whitespace);
    return text.substr(first, last - first + 1);
}

} // namespace ktt::db
