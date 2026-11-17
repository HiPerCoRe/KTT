#include <cerrno>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <variant>

#include <Database/Remote/Utility/Wire/Common.h>
#include <Utility/Logger/Logger.h>

namespace ktt::db::wire
{

std::string FingerprintToString(const size_t fingerprint)
{
    return std::to_string(fingerprint);
}

size_t FingerprintFromString(const std::string &text)
{
    // strtoull alone would accept leading whitespace, a sign ("-1" wraps around) and trailing characters.
    bool digitsOnly = !text.empty();

    for (const char c : text)
        digitsOnly = digitsOnly && c >= '0' && c <= '9';

    if (!digitsOnly)
        throw KttException("Invalid fingerprint received from the server: \"" + text + "\"");

    errno = 0;
    const unsigned long long value = std::strtoull(text.c_str(), nullptr, 10);

    if (errno == ERANGE || value > std::numeric_limits<size_t>::max())
        throw KttException("Fingerprint received from the server is out of range: " + text);

    return static_cast<size_t>(value);
}

std::string ComputeApiToString(const ComputeApi api)
{
    switch (api)
    {
    case ComputeApi::OpenCL:
        return "opencl";
    case ComputeApi::CUDA:
        return "cuda";
    case ComputeApi::Vulkan:
        return "vulkan";
    case ComputeApi::Cpp:
        return "cpp";
    }

    throw KttException("unknown compute API value " + std::to_string(static_cast<int>(api)));
}

ComputeApi ComputeApiFromString(const std::string &name)
{
    for (const auto api : {ComputeApi::OpenCL, ComputeApi::CUDA, ComputeApi::Vulkan, ComputeApi::Cpp})
    {
        if (ComputeApiToString(api) == name)
            return api;
    }

    throw KttException("unknown compute API '" + name + "' in the response");
}

// The server stores no XML, so a client configured for XML sends JSON (cpp-requirements.md R4.1.3).
OutputFormat GetWireFormat(const OutputFormat format)
{
    if (format != OutputFormat::XML)
        return format;

    static std::once_flag logged;
    std::call_once(logged, []
    {
        Logger::LogWarning("The remote database stores no XML; results saved with OutputFormat::XML are stored as JSON");
    });

    return OutputFormat::JSON;
}

std::string OutputFormatToString(const OutputFormat format)
{
    return format == OutputFormat::JSON_T4 ? "json_t4" : "json";
}

OutputFormat OutputFormatFromString(const std::string &name)
{
    if (name == "json")
        return OutputFormat::JSON;
    if (name == "json_t4")
        return OutputFormat::JSON_T4;

    throw KttException("unknown output format '" + name + "' in the response");
}

json ConfigurationToJson(const Configuration &configuration)
{
    json result = json::object();

    for (const auto &[name, value] : configuration)
        result[name] = std::visit([](const auto &alternative) { return json(alternative); }, value);

    return result;
}

static ParameterValue ParameterValueFromJson(const std::string &name, const json &value)
{
    if (value.is_boolean())
        return value.get<bool>();
    if (value.is_number_unsigned())
        return value.get<uint64_t>();
    if (value.is_number_integer())
        return value.get<int64_t>();
    if (value.is_number_float())
        return value.get<double>();
    if (value.is_string())
        return value.get<std::string>();

    throw KttException("the value of configuration parameter '" + name
        + "' in the response is not a number, boolean or string");
}

Configuration ConfigurationFromJson(const json &configuration)
{
    if (!configuration.is_object())
        throw KttException("\"configuration\" in the response is not an object");

    Configuration parameters;

    for (const auto &[name, value] : configuration.items())
        parameters.emplace(name, ParameterValueFromJson(name, value));

    return parameters;
}

} // namespace ktt::db::wire
