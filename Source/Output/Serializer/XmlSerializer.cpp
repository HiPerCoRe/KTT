#include <pugixml.hpp>

#include <Output/Serializer/XmlSerializer.h>
#include <Output/XmlConverters.h>

namespace ktt
{

void XmlSerializer::SerializeResults(const TunerMetadata& metadata, const std::vector<KernelResult>& results, std::ostream& target)
{
    pugi::xml_document document;
    pugi::xml_node root = document.append_child("TuningData");
    AppendMetadata(root, metadata);

    pugi::xml_node resultsNode = root.append_child("Results");

    for (const auto& result : results)
    {
        AppendKernelResult(resultsNode, result);
    }

    document.save(target);
}

} // namespace ktt
