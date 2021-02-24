#pragma once

#include <Output/Serializer/Serializer.h>

namespace ktt
{

class JsonSerializer : public Serializer
{
public:
    void SerializeResults(const std::vector<KernelResult>& results, const std::string& filePath) override;
};

} // namespace ktt
