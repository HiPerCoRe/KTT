#include <Output/Deserializer/JsonDeserializer.h>
#include <Output/TimeConfiguration/TimeConfiguration.h>
#include <Output/JsonConverters.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

std::vector<KernelResult> JsonDeserializer::DeserializeResults(std::istream& source)
{
    json input;
    source >> input;

    TunerMetadata metadata;
    input["Metadata"].get_to(metadata);

    if (metadata.GetTimeUnit() != TimeConfiguration::GetInstance().GetTimeUnit())
    {
        Logger::LogWarning("Loaded kernel results use different time unit than tuner");
    }

    return input["Results"].get<std::vector<KernelResult>>();
}

} // namespace ktt
