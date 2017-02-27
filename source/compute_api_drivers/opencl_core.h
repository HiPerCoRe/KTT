#pragma once

#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "opencl_context.h"
#include "opencl_device.h"
#include "opencl_platform.h"

namespace ktt
{

class OpenCLCore
{
public:
    // Constructor
    explicit OpenCLCore(const size_t platformIndex, const std::vector<size_t>& deviceIndices);

    // Platform and device retrieval methods
    static std::vector<OpenCLPlatform> getOpenCLPlatforms();
    static std::vector<OpenCLDevice> getOpenCLDevices(const OpenCLPlatform& platform);
    static void printOpenCLInfo(std::ostream& outputTarget);

private:
    // Attributes
    std::unique_ptr<OpenCLContext> context;

    // Helper methods
    static std::string getPlatformInfo(const cl_platform_id id, const cl_platform_info info);
    static std::string getDeviceInfo(const cl_device_id id, const cl_device_info info);
};

} // namespace ktt
