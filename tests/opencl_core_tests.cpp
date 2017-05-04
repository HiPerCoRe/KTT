#include "catch.hpp"

#include "../source/compute_api_driver/opencl/opencl_core.h"

TEST_CASE("Transfering data to / from buffer", "[openclCore]")
{
    ktt::OpenclCore core(0, 0);
    std::vector<float> data;
    for (size_t i = 0; i < 64; i++)
    {
        data.push_back(static_cast<float>(i));
    }

    std::unique_ptr<ktt::OpenclBuffer> buffer = core.createBuffer(ktt::ArgumentMemoryType::ReadOnly, data.size() * sizeof(float));
    REQUIRE(buffer->getType() == CL_MEM_READ_ONLY);
    REQUIRE(buffer->getBuffer() != nullptr);
    REQUIRE(buffer->getSize() == data.size() * sizeof(float));

    std::vector<float> result(data.size());
    core.updateBuffer(*buffer, data.data(), data.size() * sizeof(float));
    core.getBufferData(*buffer, result.data(), data.size() * sizeof(float));

    for (size_t i = 0; i < data.size(); i++)
    {
        REQUIRE(result.at(i) == data.at(i));
    }
}
