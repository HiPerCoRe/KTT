#pragma once

#include <string>

#include "CL/cl.h"

namespace ktt
{

class OpenCLDevice
{
public:
    explicit OpenCLDevice(const cl_device_id id, const std::string& name, const std::string& vendor):
        id(id),
        name(name),
        vendor(vendor)
    {}

    cl_device_id getId() const
    {
        return id;
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
    cl_device_id id;
    std::string name;
    std::string vendor;
};

} // namespace ktt
