#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.h>
#include <kernel_argument/kernel_argument.h>

namespace ktt
{

class VulkanPushConstant
{
public:
    explicit VulkanPushConstant(const std::vector<KernelArgument*>& scalarArguments)
    {
        range.size = 0;
        range.offset = 0;
        range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        for (const auto* argument : scalarArguments)
        {
            if (argument->getUploadType() != ArgumentUploadType::Scalar)
            {
                throw std::runtime_error("Only scalar arguments are permitted to be used as specialization constants");
            }

            const size_t dataSize = argument->getDataSizeInBytes();

            for (size_t i = 0; i < dataSize; ++i)
            {
                data.push_back(argument->getDataWithType<uint8_t>()[i]);
            }

            range.size += static_cast<uint32_t>(dataSize);
        }

        uint32_t paddingSize = 0;

        if (range.size % 4 != 0)
        {
            paddingSize = 4 - range.size % 4;
        }

        for (uint32_t i = 0; i < paddingSize; ++i)
        {
            data.push_back(0);
        }

        range.size += paddingSize;
    }

    const VkPushConstantRange& getRange() const
    {
        return range;
    }

    const std::vector<uint8_t>& getData() const
    {
        return data;
    }

private:
    VkPushConstantRange range;
    std::vector<uint8_t> data;
};

} // namespace ktt
