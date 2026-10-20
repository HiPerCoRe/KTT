#include <Database/Utility/ResultSerializationJsonT4.h>

#include <json.hpp>

#include <Output/JsonT4Converters.h>

namespace ktt::db
{

std::string SerializeResultJsonT4(const KernelResult& result, const int indent)
{
    nlohmann::json serialized;
    to_json(serialized, as_T4<const KernelResult>(result));
    return serialized.dump(indent);
}

KernelResult DeserializeResultJsonT4(const std::string& text)
{
    KernelResult result;
    as_T4<KernelResult> wrapper(result);
    from_json(nlohmann::json::parse(text), wrapper);
    return result;
}

} // namespace ktt::db
