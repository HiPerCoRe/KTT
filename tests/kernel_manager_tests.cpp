#include "catch.hpp"

#include "../source/kernel/kernel_manager.h"

TEST_CASE("kernel handling operations", "[kernelManager]")
{
    ktt::KernelManager manager;
    size_t id = manager.addKernelFromFile(std::string("test_kernel.cl"), std::string("testKernel"), ktt::DimensionVector(1024, 1, 1), ktt::DimensionVector(16, 16, 1));

    SECTION("kernel id is assigned correctly")
    {
        size_t secondId = manager.addKernelFromFile(std::string("test_kernel.cl"), std::string("testKernel"), ktt::DimensionVector(1024, 1, 1), ktt::DimensionVector(16, 16, 1));

        REQUIRE(secondId == 1);
    }

    SECTION("parameter with same name cannot be added twice")
    {
        manager.addParameter(id, ktt::KernelParameter(std::string("param"), std::vector<size_t>{1, 2, 3}));
        REQUIRE_THROWS_AS(manager.addParameter(id, ktt::KernelParameter(std::string("param"), std::vector<size_t>{2, 3})), std::runtime_error);
    }
}
