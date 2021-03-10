#include <Output/Deserializer/XmlDeserializer.h>
#include <Output/XmlConverters.h>

namespace ktt
{

std::pair<TunerMetadata, std::vector<KernelResult>> XmlDeserializer::DeserializeResults(std::istream& source)
{
    pugi::xml_document document;
    document.load(source);

    const auto root = document.child("TuningData");
    const auto metadata = ParseMetadata(root.child("Metadata"));

    std::vector<KernelResult> results;

    for (const auto result : root.child("Results").children())
    {
        results.push_back(ParseKernelResult(result));
    }

    return std::make_pair(metadata, results);
}

} // namespace ktt
