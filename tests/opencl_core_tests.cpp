#include "catch.hpp"
#include "compute_engine/opencl/opencl_core.h"
#include "kernel_argument/kernel_argument.h"

std::string programSource(std::string("")
    + "__kernel void testKernel(float number, __global float* a, __global float* b, __global float* result)\n"
    + "{\n"
    + "    int index = get_global_id(0);\n"
    + "\n"
    + "    result[index] = a[index] + b[index] + number;\n"
    + "}\n");

TEST_CASE("Working with program and kernel", "Component: OpenclCore")
{
    ktt::OpenclCore core(0, 0, ktt::RunMode::Tuning);

    auto program = core.createAndBuildProgram(programSource);
    REQUIRE(program->getSource() == programSource);

    auto kernel = core.createKernel(*program, "testKernel");
    REQUIRE(kernel->getArgumentsCount() == 0);
    REQUIRE(kernel->getKernelName() == "testKernel");

    float value = 0.0f;
    kernel->setKernelArgumentScalar(&value, sizeof(float));
    REQUIRE(kernel->getArgumentsCount() == 1);

    SECTION("Trying to build program with invalid source throws")
    {
        REQUIRE_THROWS(core.createAndBuildProgram("Invalid"));
    }
}

TEST_CASE("Working with OpenCL buffer", "Component: OpenclCore")
{
    ktt::OpenclCore core(0, 0, ktt::RunMode::Tuning);
    std::vector<float> data;
    for (size_t i = 0; i < 64; i++)
    {
        data.push_back(static_cast<float>(i));
    }

    auto argument = ktt::KernelArgument(0, data.data(), data.size(), ktt::ArgumentDataType::Float, ktt::ArgumentMemoryLocation::Device,
        ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentUploadType::Vector);

    SECTION("Transfering argument to / from device")
    {
        core.uploadArgument(argument);
        ktt::KernelArgument resultArgument = core.downloadArgument(argument.getId());

        REQUIRE(resultArgument.getDataType() == argument.getDataType());
        REQUIRE(resultArgument.getMemoryLocation() == argument.getMemoryLocation());
        REQUIRE(resultArgument.getAccessType() == argument.getAccessType());
        REQUIRE(resultArgument.getUploadType() == argument.getUploadType());
        REQUIRE(resultArgument.getDataSizeInBytes() == argument.getDataSizeInBytes());
        std::vector<float> result = resultArgument.getDataFloat();
        for (size_t i = 0; i < data.size(); i++)
        {
            REQUIRE(result.at(i) == data.at(i));
        }
    }
}
