#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.h>
#include <kernel_argument/kernel_argument.h>

namespace ktt
{

struct VulkanSpecializationInfo
{
public:
    explicit VulkanSpecializationInfo(const std::vector<KernelArgument*>& scalarArguments)
    {
        for (size_t i = 0; i < scalarArguments.size(); ++i)
        {
            const auto& argument = *scalarArguments[i];

            if (argument.getUploadType() != ArgumentUploadType::Scalar)
            {
                throw std::runtime_error("Only scalar arguments are permitted to be used as specialization constants");
            }

            VkSpecializationMapEntry entry =
            {
                static_cast<uint32_t>(i),
                static_cast<uint32_t>(specializationData.size()),
                argument.getDataSizeInBytes()
            };

            for (size_t i = 0; i < argument.getDataSizeInBytes(); ++i)
            {
                specializationData.push_back(argument.getDataWithType<uint8_t>()[i]);
            }

            mapEntries.push_back(entry);
        }

        specializationInfo.mapEntryCount = static_cast<uint32_t>(mapEntries.size()),
        specializationInfo.pMapEntries = mapEntries.data();
        specializationInfo.dataSize = specializationData.size();
        specializationInfo.pData = specializationData.data();
    }

    VkSpecializationInfo specializationInfo;
    std::vector<VkSpecializationMapEntry> mapEntries;
    std::vector<uint8_t> specializationData;
};

} // namespace ktt
