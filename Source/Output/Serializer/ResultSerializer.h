#pragma once

#include <string>
#include <vector>

#include <Api/Output/KernelResult.h>

namespace ktt
{

class ResultSerializer
{
public:
    virtual ~ResultSerializer() = default;

    virtual void SerializeResults(const std::vector<KernelResult>& results, const std::string& filePath) = 0;
};

} // namespace ktt
