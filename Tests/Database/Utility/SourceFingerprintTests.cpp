#include <catch.hpp>

// R8.3: KTT's source fingerprint of the test vectors in SourceFingerprintVectors.json (R6.4), the values the ktt-db
// backend computes as well. Not remote-specific: the fingerprint is KTT's, so this runs in every build.

#include <cstddef>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <json.hpp>

#include <Kernel/KernelDefinition.h>
#include <Utility/Fingerprint/FingerprintUtility.h>

namespace
{

#if defined(_MSC_VER)
const std::string VectorsPrefix = "";
#else
const std::string VectorsPrefix = "../";
#endif

const std::string VectorsFile = VectorsPrefix + "../Tests/Database/Utility/SourceFingerprintVectors.json";

nlohmann::json LoadVectors()
{
    std::ifstream file(VectorsFile);
    REQUIRE(file.is_open());
    return nlohmann::json::parse(file);
}

// The fingerprint as KTT computes it for a kernel: from KernelDefinitions, in the given order.
size_t FingerprintOfSources(const std::vector<std::string> &sources)
{
    std::vector<std::unique_ptr<ktt::KernelDefinition>> definitions;
    std::vector<const ktt::KernelDefinition *> pointers;

    for (size_t i = 0; i < sources.size(); ++i)
    {
        definitions.push_back(std::make_unique<ktt::KernelDefinition>(static_cast<ktt::KernelDefinitionId>(i),
            "kernel" + std::to_string(i), sources[i], ktt::DimensionVector(1), ktt::DimensionVector(1)));
        pointers.push_back(definitions.back().get());
    }

    return ktt::FingerprintUtility::GetFingerprintOfDefinitions(pointers);
}

} // namespace

TEST_CASE("The source fingerprint test vectors match KTT's fingerprint", "[Database][SourceFingerprint]")
{
    const nlohmann::json vectors = LoadVectors();
    REQUIRE(vectors.at("vectors").size() == 8);

    for (const auto &vector : vectors.at("vectors"))
    {
        const std::string name = vector.at("name").get<std::string>();
        CAPTURE(name);
        const auto sources = vector.at("sources").get<std::vector<std::string>>();
        const size_t fingerprint = FingerprintOfSources(sources);

        REQUIRE(std::to_string(fingerprint) == vector.at("fingerprint").get<std::string>());
        REQUIRE(std::stoull(vector.at("fingerprint_hex").get<std::string>(), nullptr, 16) == fingerprint);
    }
}

TEST_CASE("The source fingerprint test vectors have the expected values", "[Database][SourceFingerprint]")
{
    // The same values as the JSON file, written out, so that a change of the file or of the fingerprint fails here.
    const std::string a = "__kernel void a(__global float* x) { x[0] = 1.0f; }\n";
    const std::string b = "__global__ void b(float* x) { x[0] = 2.0f; }\n";

    REQUIRE(FingerprintOfSources({a}) == 0xe154e05befe2eef1);
    REQUIRE(FingerprintOfSources({a, b}) == 0xe5b2022076188a74);
    REQUIRE(FingerprintOfSources({b, a}) == 0xa7ad2c1ee6ad4d51);
    REQUIRE(FingerprintOfSources({""}) == 0x854121d5f7000a9d);
    REQUIRE(FingerprintOfSources({}) == 0);
    REQUIRE(FingerprintOfSources({"// \xc4\x8d" "esk\xc3\xbd text \xe2\x82\xac \xf0\x9f\x9a\x80\nint x;\n"})
        == 0x063a9e96efab4e63);
    REQUIRE(FingerprintOfSources({"int a;\r\nint b;  \r\n"}) == 0x367a76f5e0425a6f);
    REQUIRE(FingerprintOfSources({"int a;\nint b;  \n"}) == 0x115de8318401b9e4);

    // The order and the line endings matter.
    REQUIRE(FingerprintOfSources({a, b}) != FingerprintOfSources({b, a}));
    REQUIRE(FingerprintOfSources({"int a;\r\nint b;  \r\n"}) != FingerprintOfSources({"int a;\nint b;  \n"}));
}
