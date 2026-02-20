#include <Output/Serializer/JsonT4Serializer.h>
#include <Output/JsonT4Converters.h>

namespace ktt
{

void JsonT4Serializer::SerializeResults(const TunerMetadata& metadata, const std::vector<KernelResult>& results, const UserData& data, std::ostream& target)
{
    (void)data; // parameter currently unused
    json j_metadata;
    to_json(j_metadata, as_T4(metadata));
    json j_results;
    to_json(j_results, as_T4(results));
    json output
    {
        {"schema_version", "1.0.0"},
        {"metadata", j_metadata},
        {"results", j_results}
    };

    target << output.dump(2);
}

} // namespace ktt
