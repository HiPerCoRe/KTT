#pragma once

#include <string>

#include <Output/OutputFormat.h>

namespace ktt
{

std::string LoadFileToString(const std::string& filePath);
std::string GetFileExtension(const OutputFormat format);

} // namespace ktt
