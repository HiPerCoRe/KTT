#pragma once

#include <string>
#include <vector>

#include "opencl_device.h"
#include "opencl_platform.h"

namespace ktt
{

class OpenCLCore
{
public:
    // Constructor
    OpenCLCore();

    // Platform and device retrieval methods
    std::vector<OpenCLPlatform> getOpenCLPlatforms() const;
    std::vector<OpenCLDevice> getOpenCLDevices(const OpenCLPlatform& platform) const;

private:
    // Helper methods
    std::string getPlatformInfo(const cl_platform_id id, const cl_platform_info info) const;
};

} // namespace ktt
