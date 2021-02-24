#include <Output/Serializer/JsonSerializer.h>
#include <Output/JsonConverters.h>

namespace ktt
{

void JsonSerializer::SerializeResults(const TunerMetadata& metadata, const std::vector<KernelResult>& results, std::ostream& target)
{
    json output
    {
        {"Metadata", metadata},
        {"Results", results}
    };

    target << output.dump(2);
}

} // namespace ktt
