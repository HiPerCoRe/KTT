#include <Output/Deserializer/JsonT4Deserializer.h>
#include <Output/JsonT4Converters.h>

namespace ktt
{

std::pair<TunerMetadata, std::vector<KernelResult>> JsonT4Deserializer::DeserializeResults(UserData& data, std::istream& source)
{
    (void)data; // parameter currently unused
    json input;
    source >> input;
    
    const json j_metadata = input["metadata"];
    TunerMetadata metadata;
    auto metadataWrapper = as_T4(metadata);
    from_json(j_metadata, metadataWrapper);
    const json j_results = input["results"];
    std::vector<KernelResult> results;
    auto resultsWrapper = as_T4(results);
    from_json(j_results, resultsWrapper);

    return std::make_pair(metadata, results);
}

} // namespace ktt
