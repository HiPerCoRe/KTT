#include <Output/Serializer/JsonSerializer.h>
#include <Output/JsonConverters.h>

namespace ktt
{

void JsonSerializer::SerializeResults(const TunerMetadata& metadata, const std::vector<KernelResult>& results, const UserData& data,
    std::ostream& target)
{
    json output
    {
        {"Metadata", metadata},
        {"Results", results}
    };

    if (!data.empty())
    {
        output["UserData"] = data;
    }

    target << output.dump(2);
}

} // namespace ktt
