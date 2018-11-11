#include <catch.hpp>
#include <compute_engine/opencl/opencl_engine.h>
#include <kernel_argument/kernel_argument.h>

std::string programSource(std::string("")
    + "__kernel void testKernel(float number, __global float* a, __global float* b, __global float* result)\n"
    + "{\n"
    + "    int index = get_global_id(0);\n"
    + "\n"
    + "    result[index] = a[index] + b[index] + number;\n"
    + "}\n");

TEST_CASE("Working with OpenCL program and kernel", "Component: OpenCLEngine")
{
    ktt::OpenCLEngine engine(0, 0, 1);

    auto program = engine.createAndBuildProgram(programSource);
    REQUIRE(program->getSource() == programSource);

    auto kernel = std::make_unique<ktt::OpenCLKernel>(program->getProgram(), "testKernel");
    REQUIRE(kernel->getArgumentsCount() == 0);
    REQUIRE(kernel->getKernelName() == "testKernel");

    float value = 0.0f;
    kernel->setKernelArgumentScalar(&value, sizeof(float));
    REQUIRE(kernel->getArgumentsCount() == 1);

    SECTION("Trying to build program with invalid source throws")
    {
        REQUIRE_THROWS(engine.createAndBuildProgram("Invalid"));
    }
}

TEST_CASE("Working with OpenCL buffer", "Component: OpenCLEngine")
{
    ktt::OpenCLEngine engine(0, 0, 1);
    std::vector<float> data;
    for (size_t i = 0; i < 64; i++)
    {
        data.push_back(static_cast<float>(i));
    }

    auto argument = ktt::KernelArgument(0, data.data(), data.size(), sizeof(float), ktt::ArgumentDataType::Float,
        ktt::ArgumentMemoryLocation::Device, ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentUploadType::Vector, true);

    SECTION("Transfering argument to / from device")
    {
        engine.uploadArgument(argument);
        ktt::KernelArgument resultArgument = engine.downloadArgumentObject(argument.getId(), nullptr);

        REQUIRE(resultArgument.getDataType() == argument.getDataType());
        REQUIRE(resultArgument.getMemoryLocation() == argument.getMemoryLocation());
        REQUIRE(resultArgument.getAccessType() == argument.getAccessType());
        REQUIRE(resultArgument.getUploadType() == argument.getUploadType());
        REQUIRE(resultArgument.getDataSizeInBytes() == argument.getDataSizeInBytes());
        std::vector<float> result = resultArgument.getDataWithType<float>();
        for (size_t i = 0; i < data.size(); i++)
        {
            REQUIRE(result.at(i) == data.at(i));
        }
    }
}
