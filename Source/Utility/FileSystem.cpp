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

std::vector<uint8_t> LoadFileToBinary(const std::string& filePath)
{
    std::ifstream file(filePath, std::ifstream::binary);

    if (!file.is_open())
    {
        throw KttException("Unable to open file: " + filePath);
    }

    file.seekg(0, file.end);
    const auto length = static_cast<size_t>(file.tellg());
    file.seekg(0, file.beg);

    std::vector<uint8_t> result(length);
    file.read(reinterpret_cast<char*>(result.data()), length);
    return result;
}

void SaveBinaryToFile(const std::string& filePath, const std::vector<uint8_t>& output)
{
    SaveBinaryToFile(filePath, output.data(), output.size());
}

void SaveBinaryToFile(const std::string& filePath, const void* data, const size_t dataSize)
{
    std::ofstream file(filePath, std::ofstream::binary);

    if (!file.is_open())
    {
        throw KttException("Unable to open file: " + filePath);
    }

    file.write(reinterpret_cast<const char*>(data), dataSize);
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
