#include <Output/Deserializer/JsonDeserializer.h>
#include <Output/JsonConverters.h>

namespace ktt
{

std::pair<TunerMetadata, std::vector<KernelResult>> JsonDeserializer::DeserializeResults(std::istream& source)
{
    json input;
    source >> input;

    return std::make_pair(input["Metadata"].get<TunerMetadata>(), input["Results"].get<std::vector<KernelResult>>());
}

} // namespace ktt
