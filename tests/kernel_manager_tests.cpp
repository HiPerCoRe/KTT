#include "catch.hpp"
#include "api/device_info.h"
#include "kernel/kernel_manager.h"

TEST_CASE("Kernel handling operations", "[kernelManager]")
{
    ktt::KernelManager manager;
    ktt::KernelId id = manager.addKernelFromFile(std::string("../tests/test_kernel.cl"), std::string("testKernel"), ktt::DimensionVector(1024),
        ktt::DimensionVector(16, 16));

    SECTION("Kernel id is assigned correctly")
    {
        ktt::KernelId secondId = manager.addKernelFromFile(std::string("../tests/test_kernel.cl"), std::string("testKernel"),
            ktt::DimensionVector(1024), ktt::DimensionVector(16, 16));

        REQUIRE(secondId == 1);
    }

    SECTION("Kernel source is loaded correctly")
    {
        std::string source = manager.getKernel(id).getSource();
        std::string expectedSource(std::string("")
            + "__kernel void testKernel(float number, __global float* a, __global float* b, __global float* result)\n"
            + "{\n"
            + "    int index = get_global_id(0);\n"
            + "\n"
            + "    result[index] = a[index] + b[index] + number;\n"
            + "}\n");

        REQUIRE(source == expectedSource);
    }

    SECTION("Parameter with same name cannot be added twice")
    {
        manager.addParameter(id, std::string("param"), std::vector<size_t>{1, 2, 3}, ktt::ThreadModifierType::None, ktt::ThreadModifierAction::Add,
            ktt::Dimension::X);
        REQUIRE_THROWS_AS(manager.addParameter(id, std::string("param"), std::vector<size_t>{3}, ktt::ThreadModifierType::None,
            ktt::ThreadModifierAction::Add, ktt::Dimension::X), std::runtime_error);
    }
}

TEST_CASE("Kernel configuration retrieval", "[kernelManager]")
{
    ktt::KernelManager manager;
    ktt::KernelId id = manager.addKernelFromFile(std::string("../tests/test_kernel.cl"), std::string("testKernel"), ktt::DimensionVector(1024),
        ktt::DimensionVector(16, 16));
    manager.addParameter(id, std::string("param_one"), std::vector<size_t>{1, 2, 3}, ktt::ThreadModifierType::None, ktt::ThreadModifierAction::Add,
        ktt::Dimension::X);
    manager.addParameter(id, std::string("param_two"), std::vector<size_t>{5, 10}, ktt::ThreadModifierType::None, ktt::ThreadModifierAction::Add,
        ktt::Dimension::X);

    SECTION("Kernel source with defines is returned correctly")
    {
        std::vector<ktt::ParameterPair> parameterPairs;
        parameterPairs.push_back(ktt::ParameterPair("param_two", 5));
        parameterPairs.push_back(ktt::ParameterPair("param_one", 2));

        ktt::KernelConfiguration config(manager.getKernel(id).getGlobalSize(), manager.getKernel(id).getLocalSize(), parameterPairs);
        auto source = manager.getKernelSourceWithDefines(id, config);
        std::string expectedSource("#define param_one 2\n#define param_two 5\n" + manager.getKernel(id).getSource());

        REQUIRE(source == expectedSource);
    }

    SECTION("Kernel configurations are computed correctly")
    {
        ktt::DeviceInfo deviceInfo(0, std::string("Device"));
        deviceInfo.setMaxWorkGroupSize(1024);
        auto configurations = manager.getKernelConfigurations(id, deviceInfo);

        REQUIRE(configurations.size() == 6);
    }
}
