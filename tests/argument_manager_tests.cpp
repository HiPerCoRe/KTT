#include <cmath>
#include "catch.hpp"
#include "kernel_argument/argument_manager.h"

template <typename T> bool around(const T value, const T other, const T tolerance)
{
    return std::fabs(value - other) < tolerance;
}

TEST_CASE("Argument addition and retrieval", "[argumentManager]")
{
    ktt::ArgumentManager manager(ktt::RunMode::Tuning);

    std::vector<float> data{ 1.0f, 2.0f, 3.0f, 4.0f };
    ktt::ArgumentId id = manager.addArgument(data.data(), data.size(), ktt::ArgumentDataType::Float, ktt::ArgumentMemoryLocation::Device,
        ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentUploadType::Vector);

    REQUIRE(manager.getArgumentCount() == 1);
    auto argument = manager.getArgument(id);
    REQUIRE(argument.getUploadType() == ktt::ArgumentUploadType::Vector);
    REQUIRE(argument.getDataType() == ktt::ArgumentDataType::Float);
    REQUIRE(argument.getMemoryLocation() == ktt::ArgumentMemoryLocation::Device);
    REQUIRE(argument.getAccessType() == ktt::ArgumentAccessType::ReadOnly);
    REQUIRE(argument.getDataSizeInBytes() == 4 * sizeof(float));
    REQUIRE(argument.getElementSizeInBytes() == sizeof(float));

    std::vector<float> floats = argument.getDataFloat();
    REQUIRE(floats.size() == 4);

    for (size_t i = 0; i < floats.size(); i++)
    {
        REQUIRE(around(floats.at(i), data.at(i), 0.001f));
    }

    SECTION("Adding empty argument is not allowed")
    {
        REQUIRE_THROWS(manager.addArgument(data.data(), 0, ktt::ArgumentDataType::Float, ktt::ArgumentMemoryLocation::Device,
            ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentUploadType::Vector));
    }
}
