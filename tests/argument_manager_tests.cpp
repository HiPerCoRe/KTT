#include <cmath>
#include <catch.hpp>
#include <kernel_argument/argument_manager.h>

template <typename T> bool around(const T value, const T other, const T tolerance)
{
    return std::fabs(value - other) < tolerance;
}

TEST_CASE("Argument addition and retrieval", "Component: ArgumentManager")
{
    ktt::ArgumentManager manager;

    std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f};
    ktt::ArgumentId id = manager.addArgument(data.data(), data.size(), sizeof(float), ktt::ArgumentDataType::Float,
        ktt::ArgumentMemoryLocation::Device, ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentUploadType::Vector, true);

    REQUIRE(id == 0);
    REQUIRE(manager.getArgumentCount() == 1);
    ktt::KernelArgument argument = manager.getArgument(id);
    REQUIRE(argument.getUploadType() == ktt::ArgumentUploadType::Vector);
    REQUIRE(argument.getDataType() == ktt::ArgumentDataType::Float);
    REQUIRE(argument.getMemoryLocation() == ktt::ArgumentMemoryLocation::Device);
    REQUIRE(argument.getAccessType() == ktt::ArgumentAccessType::ReadOnly);
    REQUIRE(argument.getDataSizeInBytes() == 4 * sizeof(float));
    REQUIRE(argument.getElementSizeInBytes() == sizeof(float));

    std::vector<float> floats = argument.getDataWithType<float>();
    REQUIRE(floats.size() == 4);

    for (size_t i = 0; i < floats.size(); i++)
    {
        REQUIRE(around(floats.at(i), data.at(i), 0.001f));
    }

    SECTION("Adding empty argument is not allowed")
    {
        REQUIRE_THROWS(manager.addArgument(data.data(), 0, sizeof(float), ktt::ArgumentDataType::Float, ktt::ArgumentMemoryLocation::Device,
            ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentUploadType::Vector, true));
    }
}

TEST_CASE("Argument update", "Component: ArgumentManager")
{
    SECTION("Argument copies data")
    {
        ktt::ArgumentManager manager;

        std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f};
        ktt::ArgumentId id = manager.addArgument(data.data(), data.size(), sizeof(float), ktt::ArgumentDataType::Float,
            ktt::ArgumentMemoryLocation::Device, ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentUploadType::Vector, true);

        std::vector<float> newData{5.0f, 6.0f};
        manager.updateArgument(id, newData.data(), 2);

        ktt::KernelArgument argument = manager.getArgument(id);
        std::vector<float> floats = argument.getDataWithType<float>();
        REQUIRE(floats.size() == 2);

        for (size_t i = 0; i < floats.size(); i++)
        {
            REQUIRE(around(floats.at(i), newData.at(i), 0.001f));
        }

        SECTION("Original data remains unchanged")
        {
            std::vector<float> originalData{1.0f, 2.0f, 3.0f, 4.0f};
            for (size_t i = 0; i < data.size(); i++)
            {
                REQUIRE(around(data.at(i), originalData.at(i), 0.001f));
            }
        }
    }

    SECTION("Argument references data")
    {
        ktt::ArgumentManager manager;

        std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f};
        ktt::ArgumentId id = manager.addArgument(data.data(), data.size(), sizeof(float), ktt::ArgumentDataType::Float,
            ktt::ArgumentMemoryLocation::Device, ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentUploadType::Vector, false);

        std::vector<float> newData{5.0f, 6.0f};
        manager.updateArgument(id, newData.data(), 2);

        ktt::KernelArgument argument = manager.getArgument(id);
        std::vector<float> floats;
        floats.resize(argument.getNumberOfElements());
        std::memcpy(floats.data(), argument.getData(), argument.getNumberOfElements() * sizeof(float));
        REQUIRE(floats.size() == 2);

        for (size_t i = 0; i < floats.size(); i++)
        {
            REQUIRE(around(floats.at(i), newData.at(i), 0.001f));
        }

        SECTION("Original data remains unchanged")
        {
            std::vector<float> originalData{1.0f, 2.0f, 3.0f, 4.0f};
            for (size_t i = 0; i < data.size(); i++)
            {
                REQUIRE(around(data.at(i), originalData.at(i), 0.001f));
            }
        }
    }
}
