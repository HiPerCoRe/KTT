#include "catch.hpp"

#include "../source/compute_api_driver/opencl/opencl_core.h"

std::string programSource(std::string("")
    + "__kernel void testKernel(float number, __global float* a, __global float* b, __global float* result)\n"
    + "{\n"
    + "    int index = get_global_id(0);\n"
    + "\n"
    + "    result[index] = a[index] + b[index] + number;\n"
    + "}\n");

TEST_CASE("Working with program and kernel", "[openclCore]")
{
    ktt::OpenclCore core(0, 0);

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
        REQUIRE_THROWS(core.createAndBuildProgram(std::string("Invalid")));
    }
}

TEST_CASE("Transfering data to / from buffer", "[openclCore]")
{
    ktt::OpenclCore core(0, 0);
    std::vector<float> data;
    for (size_t i = 0; i < 64; i++)
    {
        data.push_back(static_cast<float>(i));
    }

    std::unique_ptr<ktt::OpenclBuffer> buffer = core.createBuffer(ktt::ArgumentMemoryType::ReadOnly, data.size() * sizeof(float), 0);
    REQUIRE(buffer->getType() == CL_MEM_READ_ONLY);
    REQUIRE(buffer->getBuffer() != nullptr);
    REQUIRE(buffer->getSize() == data.size() * sizeof(float));

    std::vector<float> result(data.size());

    core.uploadBufferData(*buffer, data.data(), data.size() * sizeof(float));
    core.downloadBufferData(*buffer, result.data(), data.size() * sizeof(float));

    for (size_t i = 0; i < data.size(); i++)
    {
        REQUIRE(result.at(i) == data.at(i));
    }
}
