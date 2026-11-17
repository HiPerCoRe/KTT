#pragma once

// Internal to the remote database client (not exported): the wire types of the ktt-db application API. Each type in
// this folder mirrors one schema of the server's OpenAPI document (ktt_db.json) and has the same name:
// - a request has to_json and a static From that maps KTT's type to it;
// - a response has from_json and a To... method that maps it to KTT's type.
// A response type holds only the fields the client reads. Its from_json throws KttException or a JSON exception for a
// malformed response; RemoteDatabase turns either into a KttException that names the operation.
//
// This header has what the types share: the scalar conversions (fingerprints, enums, configurations) and the helpers
// for optional and array fields.

#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <json.hpp>

#include <Api/KttException.h>
#include <ComputeEngine/ComputeApi.h>
#include <KttTypes.h>
#include <Output/OutputFormat.h>

namespace ktt::db::wire
{

using json = nlohmann::json;

/** @fn std::string FingerprintToString(size_t fingerprint)
 * Fingerprints are sent as decimal strings, because a 64-bit value does not fit a JSON (JavaScript) number.
 * @return The fingerprint in decimal, e.g. "14695981039346656037".
 */
std::string FingerprintToString(size_t fingerprint);

/** @fn size_t FingerprintFromString(const std::string& text)
 * Parses a fingerprint sent by the server.
 * @param text Decimal digits only, without sign or whitespace.
 * @return The fingerprint.
 * @throw KttException If text is not a decimal number that fits size_t.
 */
size_t FingerprintFromString(const std::string &text);

/** @fn std::string ComputeApiToString(ComputeApi api)
 * @return The API's name on the wire: "opencl", "cuda", "vulkan" or "cpp".
 */
std::string ComputeApiToString(ComputeApi api);

/** @fn ComputeApi ComputeApiFromString(const std::string& name)
 * @throw KttException If the name is not one ComputeApiToString returns.
 */
ComputeApi ComputeApiFromString(const std::string &name);

/** @fn OutputFormat GetWireFormat(OutputFormat format)
 * The format results are sent in. The server stores no XML, so XML becomes JSON (logged once as a warning).
 */
OutputFormat GetWireFormat(OutputFormat format);

/** @fn std::string OutputFormatToString(OutputFormat format)
 * @param format A format returned by GetWireFormat.
 * @return "json_t4" for JSON_T4, otherwise "json".
 */
std::string OutputFormatToString(OutputFormat format);

/** @fn OutputFormat OutputFormatFromString(const std::string& name)
 * @throw KttException If the name is not "json" or "json_t4".
 */
OutputFormat OutputFormatFromString(const std::string &name);

// The tuning parameters of a result, {"NAME": value}.
using Configuration = std::map<std::string, ParameterValue>;

/** @fn json ConfigurationToJson(const Configuration& configuration)
 * Numbers and booleans are sent as JSON numbers and booleans, everything else as strings.
 */
json ConfigurationToJson(const Configuration &configuration);

/** @fn Configuration ConfigurationFromJson(const json& configuration)
 * JSON carries no integer type, so a non-negative integer is read as uint64_t and a negative one as int64_t.
 * @throw KttException If configuration is not an object, or a value is not a number, boolean or string.
 */
Configuration ConfigurationFromJson(const json &configuration);

/** @fn json OptionalToJson(const std::optional<T>& value)
 * @return The value, or null when there is none.
 */
template <typename T>
json OptionalToJson(const std::optional<T> &value)
{
    return value.has_value() ? json(*value) : json(nullptr);
}

/** @fn std::optional<T> OptionalFromJson(const json& object, const char* key)
 * @return The field, or std::nullopt when it is missing or null.
 */
template <typename T>
std::optional<T> OptionalFromJson(const json &object, const char *key)
{
    const auto it = object.find(key);

    if (it == object.end() || it->is_null())
        return std::nullopt;

    return it->template get<T>();
}

/** @fn std::vector<T> ArrayFromJson(const json& object, const char* key)
 * Reads an array field, each element with T's from_json.
 * @throw KttException If the field is not an array.
 */
template <typename T>
std::vector<T> ArrayFromJson(const json &object, const char *key)
{
    const json &items = object.at(key);

    if (!items.is_array())
        throw KttException(std::string("\"") + key + "\" in the response is not an array");

    std::vector<T> result;
    result.reserve(items.size());

    for (const auto &item : items)
        result.push_back(item.template get<T>());

    return result;
}

} // namespace ktt::db::wire
