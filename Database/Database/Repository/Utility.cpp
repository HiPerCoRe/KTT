#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <random>
#include <sstream>

#include <pugixml.hpp>

#include <Api/KttException.h>
#include <Database/Repository/Utility.h>
#include <Database/Utility/ResultSerializationJsonT4.h>
#include <Output/JsonConverters.h>
#include <Output/XmlConverters.h>

namespace ktt::db
{
std::string DatabaseUtility::SqlList(const size_t size)
{
    std::string sqlList = "(";
    for (size_t i = 0; i < size; ++i)
    {
        if (i > 0)
            sqlList += ", ";
        sqlList += "?";
    }
    sqlList += ")";
    return sqlList;
}

std::string DatabaseUtility::SerializeResult(const KernelResult& result, const ktt::OutputFormat format, const int indent)
{
    switch (format)
    {
    case ktt::OutputFormat::JSON_T4:
        return SerializeResultJsonT4(result, indent);
    case ktt::OutputFormat::XML:
    {
        pugi::xml_document document;
        AppendKernelResult(document, result);
        std::ostringstream stream;
        document.save(stream);
        return stream.str();
    }
    case ktt::OutputFormat::JSON:
    default:
        return nlohmann::json(result).dump(indent);
    }
}

KernelResult DatabaseUtility::DeserializeResult(const std::string& text, const ktt::OutputFormat format)
{
    switch (format)
    {
    case ktt::OutputFormat::JSON_T4:
        return DeserializeResultJsonT4(text);
    case ktt::OutputFormat::XML:
    {
        pugi::xml_document document;
        document.load_string(text.c_str());
        return ParseKernelResult(document.child("KernelResult"));
    }
    case ktt::OutputFormat::JSON:
    default:
        return nlohmann::json::parse(text).get<KernelResult>();
    }
}
} // namespace ktt::db
