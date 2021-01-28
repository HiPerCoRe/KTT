#include <fstream>
#include <sstream>

#include <Utility/ErrorHandling/KttException.h>
#include <Utility/FileSystem.h>

namespace ktt
{

std::string LoadFileToString(const std::string& filePath)
{
    std::ifstream file(filePath);

    if (!file.is_open())
    {
        throw KttException("Unable to open file: " + filePath);
    }

    std::stringstream stream;
    stream << file.rdbuf();
    return stream.str();
}

} // namespace ktt
