// RemoteDatabase: the kernel source code upload (cpp-requirements.md R6) and the automatic upload after a save
// (R4.1.8, R6.3).

#include <cstddef>
#include <exception>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <json.hpp>

#include <Api/KttException.h>
#include <Database/Remote/RemoteDatabase.h>
#include <Database/Remote/Utility/RemoteCall.h>
#include <Database/Remote/Utility/RemoteInternal.h>
#include <Database/Remote/Utility/Wire/Source.h>
#include <Tuner.h>
#include <Utility/Fingerprint/FingerprintUtility.h>
#include <Utility/Logger/Logger.h>

namespace ktt::db
{

namespace
{

using json = nlohmann::json;

// The server's limits for PUT /source/{fingerprint}/code (endpoints.md). Sizes are UTF-8 bytes.
constexpr size_t MinDefinitions = 1;
constexpr size_t MaxDefinitions = 32;
constexpr size_t MaxSourceBytes = 1024 * 1024;
constexpr size_t MaxTotalSourceBytes = 4 * 1024 * 1024;
constexpr size_t MaxNameBytes = 1024;
constexpr size_t MaxBodyBytes = 5 * 1024 * 1024;

// The byte offset of the first invalid UTF-8 sequence in text, or std::nullopt if the text is valid UTF-8. Strict, as
// the server checks it: no overlong forms, no surrogates (U+D800 to U+DFFF) and nothing above U+10FFFF.
std::optional<size_t> FindInvalidUtf8(const std::string &text)
{
    const auto byte = [&text](const size_t index) { return static_cast<unsigned char>(text[index]); };

    for (size_t i = 0; i < text.size();)
    {
        const unsigned char lead = byte(i);

        if (lead < 0x80)
        {
            ++i;
            continue;
        }

        size_t length = 0;
        // The range of the second byte; the lead byte narrows it to rule out overlong forms, surrogates and values
        // above U+10FFFF.
        unsigned char low = 0x80;
        unsigned char high = 0xBF;

        if (lead >= 0xC2 && lead <= 0xDF)
        {
            length = 2;
        }
        else if (lead >= 0xE0 && lead <= 0xEF)
        {
            length = 3;
            low = lead == 0xE0 ? 0xA0 : low;
            high = lead == 0xED ? 0x9F : high;
        }
        else if (lead >= 0xF0 && lead <= 0xF4)
        {
            length = 4;
            low = lead == 0xF0 ? 0x90 : low;
            high = lead == 0xF4 ? 0x8F : high;
        }
        else
        {
            return i;
        }

        if (text.size() - i < length || byte(i + 1) < low || byte(i + 1) > high)
            return i;

        for (size_t k = 2; k < length; ++k)
        {
            if (byte(i + k) < 0x80 || byte(i + k) > 0xBF)
                return i;
        }

        i += length;
    }

    return std::nullopt;
}

void CheckText(const std::string &operation, const size_t position, const std::string &field, const std::string &text)
{
    const std::string what = "the " + field + " of definition " + std::to_string(position);

    if (const auto offset = FindInvalidUtf8(text); offset.has_value())
        throw KttException(operation + ": " + what + " is not valid UTF-8 (at byte " + std::to_string(*offset) + ")");

    if (const auto offset = text.find('\0'); offset != std::string::npos)
    {
        throw KttException(operation + ": " + what + " contains the character U+0000 (at byte "
            + std::to_string(offset) + "), which the server cannot store");
    }
}

// The checks of the server, in its order: the count, then the sizes, then the text. The positions are 0-based.
void CheckDefinitions(const std::string &operation, const std::vector<SourceDefinition> &definitions)
{
    if (definitions.size() < MinDefinitions || definitions.size() > MaxDefinitions)
    {
        throw KttException(operation + ": the code must have " + std::to_string(MinDefinitions) + " to "
            + std::to_string(MaxDefinitions) + " definitions, got " + std::to_string(definitions.size()));
    }

    size_t totalBytes = 0;

    for (size_t position = 0; position < definitions.size(); ++position)
    {
        const SourceDefinition &definition = definitions[position];

        if (definition.source.size() > MaxSourceBytes)
        {
            throw KttException(operation + ": the source of definition " + std::to_string(position) + " has "
                + std::to_string(definition.source.size()) + " bytes, more than " + std::to_string(MaxSourceBytes));
        }

        if (definition.name.size() > MaxNameBytes)
        {
            throw KttException(operation + ": the name of definition " + std::to_string(position) + " has "
                + std::to_string(definition.name.size()) + " bytes, more than " + std::to_string(MaxNameBytes));
        }

        CheckText(operation, position, "name", definition.name);
        CheckText(operation, position, "source", definition.source);
        totalBytes += definition.source.size();
    }

    if (totalBytes > MaxTotalSourceBytes)
    {
        throw KttException(operation + ": the sources have " + std::to_string(totalBytes) + " bytes together, more than "
            + std::to_string(MaxTotalSourceBytes));
    }
}

} // namespace

size_t GetSourceFingerprint(const std::vector<SourceDefinition> &definitions)
{
    // The loop of FingerprintUtility::GetFingerprintOfDefinitions, over the sources instead of KernelDefinitions.
    size_t fingerprint = 0;

    for (const auto &definition : definitions)
        fingerprint = FingerprintUtility::HashFunction(fingerprint, FingerprintUtility::HashString(definition.source));

    return fingerprint;
}

bool RemoteDatabase::SaveSourceCode(const size_t sourceFingerprint, const std::vector<SourceDefinition> &definitions) const
{
    static const std::string Operation = "RemoteDatabase::SaveSourceCode";
    CheckDefinitions(Operation, definitions);

    // A mismatch is a bug in the caller (other or reordered definitions, or changed text), and the server would reject
    // the code anyway.
    if (const size_t fingerprint = GetSourceFingerprint(definitions); fingerprint != sourceFingerprint)
    {
        throw KttException(Operation + ": the sources have the source fingerprint " + wire::FingerprintToString(fingerprint)
            + ", not " + wire::FingerprintToString(sourceFingerprint) + "; pass the kernel's definitions unchanged and in their "
            "order (Tuner::GetKernelSources)");
    }

    long status = 0;

    TranslateRemote(Operation, *m_Client, [&]
    {
        const json body = wire::SaveSourceCodeRequest::From(definitions);

        // JSON escaping (quotes, backslashes, control characters) makes the body larger than the sources.
        if (const size_t bodyBytes = body.dump().size(); bodyBytes > MaxBodyBytes)
        {
            throw KttException(Operation + ": the request body has " + std::to_string(bodyBytes)
                + " bytes with its JSON escaping, more than the server accepts (" + std::to_string(MaxBodyBytes) + ")");
        }

        // The code is immutable on the server, so a repeated PUT is safe: it answers 200 and changes nothing.
        CallRemote(*m_Client, Operation, "PUT", "/source/" + wire::FingerprintToString(sourceFingerprint) + "/code", &body,
            false, &status);
    });

    return status == 201;
}

void UploadMissingSourceCode(const RemoteDatabase &database, const SourceCodeProvider &provider,
    const size_t sourceFingerprint, const std::optional<bool> sourceHasCode)
{
    if (!provider)
        return;

    const std::string source = "source " + wire::FingerprintToString(sourceFingerprint);
    if (!sourceHasCode.has_value())
    {
        Logger::LogWarning("RemoteDatabase::SaveResults: the results were saved, but the response does not say whether "
            "the server has the code of " + source + ", so it is not uploaded");
        return;
    }

    if (*sourceHasCode)
        return;

    try
    {
        const auto definitions = provider(sourceFingerprint);

        if (!definitions.has_value())
        {
            Logger::LogInfo("RemoteDatabase::SaveResults: the server has no code for " + source
                + ", and the source-code provider has none to upload");
            return;
        }

        if (database.SaveSourceCode(sourceFingerprint, *definitions))
            Logger::LogInfo("RemoteDatabase::SaveResults: uploaded the code of " + source);
    }
    catch (const std::exception &error)
    {
        Logger::LogWarning("RemoteDatabase::SaveResults: the results were saved, but uploading the code of " + source
            + " failed: " + error.what());
    }
    catch (...)
    {
        Logger::LogWarning("RemoteDatabase::SaveResults: the results were saved, but uploading the code of " + source
            + " failed with an unknown exception");
    }
}

SourceCodeProvider MakeSourceCodeProvider(const Tuner &tuner, std::vector<KernelId> kernels)
{
    return [&tuner, kernels = std::move(kernels)](const size_t sourceFingerprint)
        -> std::optional<std::vector<SourceDefinition>>
    {
        for (const KernelId kernel : kernels)
        {
            // Empty (with an error logged by the tuner) for a kernel that no longer exists.
            auto definitions = tuner.GetKernelSources(kernel);

            if (!definitions.empty() && GetSourceFingerprint(definitions) == sourceFingerprint)
                return definitions;
        }

        return std::nullopt;
    };
}

} // namespace ktt::db
