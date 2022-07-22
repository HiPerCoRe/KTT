#pragma once

#include <string>

#include <Output/OutputFormat.h>

namespace ktt
{

std::string LoadFileToString(const std::string& filePath);
void SaveStringToFile(const std::string& filePath, const std::string& output);
std::string GetFileExtension(const OutputFormat format);

} // namespace ktt
