#include "catch.hpp"

#include "../source/kernel_argument/argument_manager.h"

bool around(const float value, const float other, const float tolerance)
{
    return value - tolerance < other && value + tolerance > other;
}

TEST_CASE("Argument addition and retrieval", "[argumentManager]")
{
    ktt::ArgumentManager manager;

    std::vector<float> data{ 1.0f, 2.0f, 3.0f, 4.0f };
    size_t id = manager.addArgument(data, ktt::ArgumentMemoryType::READ_ONLY);

    REQUIRE(manager.getArgumentCount() == 1);
    auto argument = manager.getArgument(id);
    REQUIRE(argument.getArgumentQuantity() == ktt::ArgumentQuantity::Vector);
    REQUIRE(argument.getArgumentDataType() == ktt::ArgumentDataType::Float);
    REQUIRE(argument.getDataSize() == 4 * sizeof(float));

    std::vector<float> floats = argument.getDataFloat();
    REQUIRE(floats.size() == 4);

    for (size_t i = 0; i < floats.size(); i++)
    {
        REQUIRE(around(floats.at(i), data.at(i), 0.001f));
    }
}
