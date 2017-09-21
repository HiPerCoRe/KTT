#pragma once

#include <string>

#include "CL/cl.h"

namespace ktt
{

class OpenclPlatform
{
public:
    explicit OpenclPlatform(const cl_platform_id id, const std::string& name) :
        id(id),
        name(name)
    {}

    cl_platform_id getId() const
    {
        return id;
    }

    std::string getName() const
    {
        return name;
    }

private:
    cl_platform_id id;
    std::string name;
};

} // namespace ktt
