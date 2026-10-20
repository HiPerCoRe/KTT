#include <catch.hpp>

#include <Api/KttException.h>
#include <ComputeEngine/Cuda/CudaEngine.h>
#include <KernelArgument/KernelArgument.h>
#include <Utility/NumericalUtilities.h>

TEST_CASE("Working with CUDA buffer", "CudaEngine")
{
    ktt::CudaEngine engine(0, 1);
    std::vector<float> data;

    for (size_t i = 0; i < 64; ++i)
    {
        data.push_back(static_cast<float>(i));
    }

    ktt::KernelArgument argument(ktt::ArgumentId("0"), sizeof(float), ktt::ArgumentDataType::Float, ktt::ArgumentMemoryLocation::Device,
        ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryType::Vector, ktt::ArgumentManagementType::Framework);
    argument.SetOwnedData(data.data(), data.size() * sizeof(float));

    SECTION("Transfering argument to / from device")
    {
        const auto uploadAction = engine.UpdateArgument(argument, engine.GetDefaultQueue());
        engine.WaitForTransferAction(uploadAction);

        std::vector<float> result(64);
        const auto downloadAction = engine.DownloadArgument(argument.GetId(), engine.GetDefaultQueue(), result.data(),
            result.size() * sizeof(float));
        engine.WaitForTransferAction(downloadAction);

        for (size_t i = 0; i < data.size(); ++i)
        {
            REQUIRE(ktt::FloatEquals(result[i], data[i], 0.001f));
        }
    }

    SECTION("Updating argument buffer with new data")
    {
        const auto uploadAction = engine.UpdateArgument(argument, engine.GetDefaultQueue());
        engine.WaitForTransferAction(uploadAction);

        std::vector<float> newData(64, 5.0f);
        const auto updateAction = engine.UpdateArgument(argument, engine.GetDefaultQueue(), newData.data(),
            newData.size() * sizeof(float));
        engine.WaitForTransferAction(updateAction);

        std::vector<float> result(64);
        const auto downloadAction = engine.DownloadArgument(argument.GetId(), engine.GetDefaultQueue(), result.data(),
            result.size() * sizeof(float));
        engine.WaitForTransferAction(downloadAction);

        for (size_t i = 0; i < result.size(); ++i)
        {
            REQUIRE(ktt::FloatEquals(result[i], newData[i], 0.001f));
        }
    }

    SECTION("Updating argument buffer from kernel argument data")
    {
        const auto uploadAction = engine.UpdateArgument(argument, engine.GetDefaultQueue());
        engine.WaitForTransferAction(uploadAction);

        std::vector<float> newData(64, 7.0f);
        argument.SetOwnedData(newData.data(), newData.size() * sizeof(float));

        const auto updateAction = engine.UpdateArgument(argument, engine.GetDefaultQueue());
        engine.WaitForTransferAction(updateAction);

        std::vector<float> result(64);
        const auto downloadAction = engine.DownloadArgument(argument.GetId(), engine.GetDefaultQueue(), result.data(),
            result.size() * sizeof(float));
        engine.WaitForTransferAction(downloadAction);

        for (size_t i = 0; i < result.size(); ++i)
        {
            REQUIRE(ktt::FloatEquals(result[i], newData[i], 0.001f));
        }
    }

    SECTION("Updating non-vector argument throws")
    {
        std::vector<float> scalarData(1, 1.0f);
        ktt::KernelArgument scalarArgument(ktt::ArgumentId("1"), sizeof(float), ktt::ArgumentDataType::Float,
            ktt::ArgumentMemoryLocation::Undefined, ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryType::Scalar,
            ktt::ArgumentManagementType::Framework);
        scalarArgument.SetOwnedData(scalarData.data(), scalarData.size() * sizeof(float));

        REQUIRE_THROWS_AS(engine.UpdateArgument(scalarArgument, engine.GetDefaultQueue()), ktt::KttException);
    }

    SECTION("Zeroing argument buffer")
    {
        const auto uploadAction = engine.UpdateArgument(argument, engine.GetDefaultQueue());
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
        ktt::KernelArgument charArgument(ktt::ArgumentId("1"), sizeof(char), ktt::ArgumentDataType::Char, ktt::ArgumentMemoryLocation::Device,
            ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryType::Vector, ktt::ArgumentManagementType::Framework);
        charArgument.SetOwnedData(charData.data(), charData.size());

        const auto uploadAction = engine.UpdateArgument(charArgument, engine.GetDefaultQueue());
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
