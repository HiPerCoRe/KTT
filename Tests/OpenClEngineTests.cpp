#include <catch.hpp>

#include <ComputeEngine/OpenCl/OpenClEngine.h>
#include <KernelArgument/KernelArgument.h>
#include <Utility/NumericalUtilities.h>

TEST_CASE("Working with OpenCL buffer", "OpenClEngine")
{
    ktt::OpenClEngine engine(0, 0, 1);
    std::vector<float> data;

    for (size_t i = 0; i < 64; ++i)
    {
        data.push_back(static_cast<float>(i));
    }

    ktt::KernelArgument argument(0, sizeof(float), ktt::ArgumentDataType::Float, ktt::ArgumentMemoryLocation::Device,
        ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryType::Vector, ktt::ArgumentManagementType::Framework);
    argument.SetOwnedData(data.data(), data.size() * sizeof(float));

    SECTION("Transfering argument to / from device")
    {
        const auto uploadAction = engine.UploadArgument(argument, engine.GetDefaultQueue());
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
}
