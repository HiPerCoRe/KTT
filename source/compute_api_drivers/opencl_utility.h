#pragma once

#include <string>

#include "CL/cl.h"

namespace ktt
{

std::string getOpenCLEnumName(const cl_int value);
void checkOpenCLError(const cl_int value);
void checkOpenCLError(const cl_int value, const std::string& message);

} // namespace ktt
