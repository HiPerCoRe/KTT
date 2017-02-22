#include "catch.hpp"

#include "../source/kernel/kernel_manager.h"

TEST_CASE("kernel handling operations", "[kernelManager]")
{
    ktt::KernelManager manager;
    size_t id = manager.addKernelFromFile(std::string("../tests/test_kernel.cl"), std::string("testKernel"), ktt::DimensionVector(1024, 1, 1),
        ktt::DimensionVector(16, 16, 1));

    SECTION("kernel id is assigned correctly")
    {
        size_t secondId = manager.addKernelFromFile(std::string("../tests/test_kernel.cl"), std::string("testKernel"),
            ktt::DimensionVector(1024, 1, 1), ktt::DimensionVector(16, 16, 1));

        REQUIRE(secondId == 1);
    }

    SECTION("kernel source is loaded correctly")
    {
        std::string source = manager.getKernel(id)->getSource();
        std::string expectedSource(std::string("")
            + "__kernel void testKernel(float number, __global float* a, __global float* b, __global float* result)\n"
            + "{\n"
            + "    int index = get_global_id(0);\n"
            + "\n"
            + "    result[index] = a[index] + b[index] + number;\n"
            + "}\n");

        REQUIRE(source == expectedSource);
    }

    SECTION("parameter with same name cannot be added twice")
    {
        manager.addParameter(id, ktt::KernelParameter(std::string("param"), std::vector<size_t>{1, 2, 3}));
        REQUIRE_THROWS_AS(manager.addParameter(id, ktt::KernelParameter(std::string("param"), std::vector<size_t>{2, 3})), std::runtime_error);
    }
}

TEST_CASE("kernel configuration retrieval", "[kernelManager]")
{
    ktt::KernelManager manager;
    size_t id = manager.addKernelFromFile(std::string("test_kernel.cl"), std::string("testKernel"), ktt::DimensionVector(1024, 1, 1),
        ktt::DimensionVector(16, 16, 1));
    manager.addParameter(id, ktt::KernelParameter(std::string("param_one"), std::vector<size_t>{1, 2, 3}));
    manager.addParameter(id, ktt::KernelParameter(std::string("param_two"), std::vector<size_t>{5, 10}));

    SECTION("kernel source with defines is returned correctly")
    {
        std::vector<ktt::ParameterValue> values;
        values.push_back(ktt::ParameterValue("param_two", 5));
        values.push_back(ktt::ParameterValue("param_one", 2));

        ktt::KernelConfiguration config(manager.getKernel(id)->getGlobalSize(), manager.getKernel(id)->getLocalSize(), values);
        auto source = manager.getKernelSourceWithDefines(id, config);
        std::string expectedSource("#define param_one 2\n#define param_two 5\n" + manager.getKernel(id)->getSource());

        REQUIRE(source == expectedSource);
    }

    SECTION("kernel configurations are computed correctly")
    {
        auto configurations = manager.getKernelConfigurations(id);

        REQUIRE(configurations.size() == 6);
    }
}
