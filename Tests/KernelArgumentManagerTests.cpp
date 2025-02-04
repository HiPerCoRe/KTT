#include <catch.hpp>

#include <Api/KttException.h>
#include <KernelArgument/KernelArgumentManager.h>
#include <Utility/NumericalUtilities.h>

TEST_CASE("Argument addition, retrieval and delete", "KernelArgumentManager")
{
    ktt::KernelArgumentManager manager;

    std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f};
    const ktt::ArgumentId id = manager.AddArgumentWithOwnedData(sizeof(float), ktt::ArgumentDataType::Float,
        ktt::ArgumentMemoryLocation::Device, ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryType::Vector,
        ktt::ArgumentManagementType::Framework, data.data(), data.size() * sizeof(float));

    REQUIRE(id == "0");
    ktt::KernelArgument& argument = manager.GetArgument(id);
    REQUIRE(argument.GetMemoryType() == ktt::ArgumentMemoryType::Vector);
    REQUIRE(argument.GetDataType() == ktt::ArgumentDataType::Float);
    REQUIRE(argument.GetMemoryLocation() == ktt::ArgumentMemoryLocation::Device);
    REQUIRE(argument.GetAccessType() == ktt::ArgumentAccessType::ReadOnly);
    REQUIRE(argument.GetNumberOfElements() == 4);
    REQUIRE(argument.GetElementSize() == sizeof(float));
    REQUIRE(argument.GetDataSize() == 4 * sizeof(float));

    const float* floats = argument.GetDataWithType<float>();
    const size_t floatsSize = argument.GetNumberOfElementsWithType<float>();
    REQUIRE(floatsSize == 4);

    for (size_t i = 0; i < floatsSize; ++i)
    {
        REQUIRE(ktt::FloatEquals(floats[i], data[i], 0.001f));
    }

    manager.RemoveArgument(id);
    REQUIRE_THROWS(manager.GetArgument(id));

    SECTION("Adding zero-size vector argument is not allowed")
    {
        REQUIRE_THROWS_AS(manager.AddArgumentWithOwnedData(sizeof(float), ktt::ArgumentDataType::Float,
            ktt::ArgumentMemoryLocation::Device, ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryType::Vector,
            ktt::ArgumentManagementType::Framework, data.data(), 0), ktt::KttException);
    }
}
