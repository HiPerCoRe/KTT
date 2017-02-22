#pragma once

#include <string>

#include "CL/cl.h"

namespace ktt
{

class OpenCLPlatform
{
public:
    explicit OpenCLPlatform(const cl_platform_id id, const std::string& openCLVersion, const std::string& name, const std::string& vendor):
        id(id),
        openCLVersion(openCLVersion),
        name(name),
        vendor(vendor)
    {}

    cl_platform_id getId() const
    {
        return id;
    }

    std::string getOpenCLVersion() const
    {
        return openCLVersion;
    }

    std::string getName() const
    {
        return name;
    }

    std::string getVendor() const
    {
        return vendor;
    }

private:
    cl_platform_id id;
    std::string openCLVersion;
    std::string name;
    std::string vendor;
};

} // namespace ktt
