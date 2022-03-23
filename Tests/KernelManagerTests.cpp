#include <catch.hpp>

#include <Api/KttException.h>
#include <Kernel/KernelManager.h>

#if defined(_MSC_VER)
const std::string kernelPrefix = "";
#else
const std::string kernelPrefix = "../";
#endif

const std::string kernelSource = kernelPrefix + "../Tests/Kernels/SimpleOpenClKernel.cl";

TEST_CASE("Kernel handling operations", "KernelManager")
{
    ktt::KernelArgumentManager argumentManager;
    ktt::KernelManager manager(argumentManager);
    
    const ktt::KernelDefinitionId definition = manager.AddKernelDefinitionFromFile("simpleKernel", kernelSource,
        ktt::DimensionVector(1024), ktt::DimensionVector(8, 8));
    REQUIRE(definition == 0);

    SECTION("Kernel definition id is assigned correctly")
    {
        const ktt::KernelDefinitionId definition2 = manager.AddKernelDefinitionFromFile("simpleKernel2", kernelSource,
            ktt::DimensionVector(1024), ktt::DimensionVector(8, 8));
        REQUIRE(definition2 == 1);
    }

    SECTION("Kernel source is loaded correctly")
    {
        const auto& source = manager.GetDefinition(definition).GetSource();

        std::string expectedSource(std::string("")
            + "__kernel void simpleKernel(float number, __global float* a, __global float* b, __global float* result)\n"
            + "{\n"
            + "    int index = get_global_id(0);\n"
            + "\n"
            + "    result[index] = a[index] + b[index] + number;\n"
            + "}\n");

        REQUIRE(source == expectedSource);
    }

    const ktt::KernelId kernel = manager.CreateKernel("kernel", {definition});
    REQUIRE(kernel == 0);

    SECTION("Parameter with same name cannot be added twice")
    {
        manager.AddParameter(kernel, "param", std::vector<ktt::ParameterValue>{1LLU, 2LLU, 3LLU}, "");
        REQUIRE_THROWS_AS(manager.AddParameter(kernel, "param", std::vector<ktt::ParameterValue>{3LLU}, ""), ktt::KttException);
    }
}

TEST_CASE("Adding preprocessor definitions to kernel source", "KernelManager")
{
    ktt::KernelArgumentManager argumentManager;
    ktt::KernelManager manager(argumentManager);

    const ktt::KernelDefinitionId definition = manager.AddKernelDefinitionFromFile("simpleKernel", kernelSource,
        ktt::DimensionVector(1024), ktt::DimensionVector(8, 8));
    const ktt::KernelId kernel = manager.CreateKernel("kernel", {definition});

    manager.AddParameter(kernel, "param_one", std::vector<ktt::ParameterValue>{1LLU, 2LLU, 3LLU}, "");
    manager.AddParameter(kernel, "param_two", std::vector<ktt::ParameterValue>{5LLU, 10LLU}, "");

    SECTION("Kernel configuration prefix is generated correctly")
    {
        std::vector<ktt::ParameterPair> parameterPairs;
        parameterPairs.emplace_back("param_two", static_cast<uint64_t>(5));
        parameterPairs.emplace_back("param_one", static_cast<uint64_t>(2));

        ktt::KernelConfiguration configuration(parameterPairs);
        const std::string prefix = configuration.GeneratePrefix();
        std::string expectedPrefix("#define param_two 5\n#define param_one 2\n");

        REQUIRE(prefix == expectedPrefix);
    }
}
