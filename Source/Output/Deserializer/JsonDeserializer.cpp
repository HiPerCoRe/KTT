#include <Output/Deserializer/JsonDeserializer.h>
#include <Output/JsonConverters.h>

namespace ktt
{

std::pair<TunerMetadata, std::vector<KernelResult>> JsonDeserializer::DeserializeResults(UserData& data, std::istream& source)
{
    json input;
    source >> input;

    if (input.contains("UserData"))
    {
        data = input["UserData"].get<UserData>();
    }

    return std::make_pair(input["Metadata"].get<TunerMetadata>(), input["Results"].get<std::vector<KernelResult>>());
}

} // namespace ktt
