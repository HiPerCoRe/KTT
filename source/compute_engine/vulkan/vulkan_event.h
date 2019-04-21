#pragma once

#include <memory>
#include <compute_engine/vulkan/vulkan_fence.h>
#include <compute_engine/vulkan/vulkan_utility.h>
#include <ktt_types.h>

namespace ktt
{

class VulkanEvent
{
public:
    explicit VulkanEvent(VkDevice device, const EventId id, const bool validFlag) :
        id(id),
        kernelName(""),
        fence(nullptr),
        validFlag(validFlag),
        overhead(0)
    {
        if (validFlag)
        {
            fence = std::make_unique<VulkanFence>(device);
        }
    }

    explicit VulkanEvent(VkDevice device, const EventId id, const std::string& kernelName, const uint64_t kernelLaunchOverhead) :
        id(id),
        kernelName(kernelName),
        fence(std::make_unique<VulkanFence>(device)),
        validFlag(true),
        overhead(kernelLaunchOverhead)
    {}

    EventId getId() const
    {
        return id;
    }

    const std::string& getKernelName() const
    {
        return kernelName;
    }

    const VulkanFence& getFence() const
    {
        if (!isValid())
        {
            throw std::runtime_error("Valid Vulkan fence cannot be accessed by invalid Vulkan events");
        }

        return *fence.get();
    }

    VulkanFence& getFence()
    {
        if (!isValid())
        {
            throw std::runtime_error("Valid Vulkan fence cannot be accessed by invalid Vulkan events");
        }

        return *fence.get();
    }

    bool isValid() const
    {
        return validFlag;
    }

    uint64_t getOverhead() const
    {
        return overhead;
    }

    void wait()
    {
        if (!isValid())
        {
            throw std::runtime_error("Valid Vulkan fence cannot be accessed by invalid Vulkan events");
        }

        fence->wait();
    }

private:
    EventId id;
    std::string kernelName;
    std::unique_ptr<VulkanFence> fence;
    bool validFlag;
    uint64_t overhead;
};

} // namespace ktt
