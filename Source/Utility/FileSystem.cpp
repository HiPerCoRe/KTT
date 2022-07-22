#include <fstream>
#include <sstream>

#include <Api/KttException.h>
#include <Utility/ErrorHandling/Assert.h>
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

void SaveStringToFile(const std::string& filePath, const std::string& output)
{
    std::ofstream file(filePath);

    if (!file.is_open())
    {
        throw KttException("Unable to open file: " + filePath);
    }

    file << output;
}

std::string GetFileExtension(const OutputFormat format)
{
    switch (format)
    {
    case OutputFormat::JSON:
        return ".json";
    case OutputFormat::XML:
        return ".xml";
    default:
        KttError("Unhandled output format value");
        return "";
    }
}

} // namespace ktt
