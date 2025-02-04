#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <Output/OutputFormat.h>

namespace ktt
{

std::string LoadFileToString(const std::string& filePath);
void SaveStringToFile(const std::string& filePath, const std::string& output);

std::vector<uint8_t> LoadFileToBinary(const std::string& filePath);
void SaveBinaryToFile(const std::string& filePath, const std::vector<uint8_t>& output);
void SaveBinaryToFile(const std::string& filePath, const void* data, const size_t dataSize);

std::string GetFileExtension(const OutputFormat format);

} // namespace ktt
