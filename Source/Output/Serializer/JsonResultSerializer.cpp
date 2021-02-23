#include <fstream>

#include <Api/KttException.h>
#include <Output/Serializer/JsonResultSerializer.h>
#include <Output/JsonConverters.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

void JsonResultSerializer::SerializeResults(const std::vector<KernelResult>& results, const std::string& filePath)
{
    const std::string file = filePath + ".json";
    Logger::LogInfo("Saving kernel results in JSON format to file: " + file);
    std::ofstream outputFile(file);

    if (!outputFile.is_open())
    {
        throw KttException("Unable to open file: " + file);
    }

    json output{"Results", results};
    outputFile << output.dump(2);
}

} // namespace ktt
