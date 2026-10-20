#include <catch.hpp>

#include <Api/KttException.h>
#include <ComputeEngine/Vulkan/VulkanEngine.h>
#include <KernelArgument/KernelArgument.h>

TEST_CASE("Working with Vulkan buffer", "VulkanEngine")
{
    ktt::VulkanEngine engine(0, 1);
    std::vector<float> data;

    for (size_t i = 0; i < 64; ++i)
    {
        data.push_back(static_cast<float>(i));
    }

    ktt::KernelArgument argument(ktt::ArgumentId("0"), sizeof(float), ktt::ArgumentDataType::Float,
        ktt::ArgumentMemoryLocation::Device, ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryType::Vector,
        ktt::ArgumentManagementType::Framework);
    argument.SetOwnedData(data.data(), data.size() * sizeof(float));

    SECTION("Zeroing argument buffer")
    {
        const auto uploadAction = engine.UploadArgument(argument, engine.GetDefaultQueue());
        engine.WaitForTransferAction(uploadAction);

        engine.ZeroBuffer(argument.GetId(), engine.GetDefaultQueue());

        std::vector<float> result(64);
        const auto downloadAction = engine.DownloadArgument(argument.GetId(), engine.GetDefaultQueue(), result.data(),
            result.size() * sizeof(float));
        engine.WaitForTransferAction(downloadAction);

        for (size_t i = 0; i < result.size(); ++i)
        {
            REQUIRE(result[i] == 0.0f);
        }
    }

    SECTION("Zeroing buffer with size not divisible by four")
    {
        std::vector<char> charData(10, 1);
        ktt::KernelArgument charArgument(ktt::ArgumentId("1"), sizeof(char), ktt::ArgumentDataType::Char,
            ktt::ArgumentMemoryLocation::Device, ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryType::Vector,
            ktt::ArgumentManagementType::Framework);
        charArgument.SetOwnedData(charData.data(), charData.size());

        const auto uploadAction = engine.UploadArgument(charArgument, engine.GetDefaultQueue());
        engine.WaitForTransferAction(uploadAction);

        engine.ZeroBuffer(charArgument.GetId(), engine.GetDefaultQueue());

        std::vector<char> result(10);
        const auto downloadAction = engine.DownloadArgument(charArgument.GetId(), engine.GetDefaultQueue(), result.data(),
            result.size());
        engine.WaitForTransferAction(downloadAction);

        for (const auto value : result)
        {
            REQUIRE(value == 0);
        }
    }

    SECTION("Zeroing missing buffer throws")
    {
        REQUIRE_THROWS_AS(engine.ZeroBuffer("invalid", engine.GetDefaultQueue()), ktt::KttException);
    }
}
