#pragma once

#include <string>

namespace ktt
{

bool StartsWith(const std::string& target, const std::string& prefix);

// Compute API methods (e.g., CUDA, OpenCL) which return string in char* format often add terminating \0 character to the end.
// KTT stores the string in std::string buffer where the terminating character is unwanted. This method removes that character.
void RemoveTrailingZero(std::string& target);

} // namespace ktt
