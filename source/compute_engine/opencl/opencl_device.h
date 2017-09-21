#pragma once

#include <string>

#include "CL/cl.h"

namespace ktt
{

class OpenclDevice
{
public:
    explicit OpenclDevice(const cl_device_id id, const std::string& name) :
        id(id),
        name(name)
    {}

    cl_device_id getId() const
    {
        return id;
    }

    std::string getName() const
    {
        return name;
    }

private:
    cl_device_id id;
    std::string name;
};

} // namespace ktt
