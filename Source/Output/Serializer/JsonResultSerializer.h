#pragma once

#include <Output/Serializer/ResultSerializer.h>

namespace ktt
{

class JsonResultSerializer : public ResultSerializer
{
public:
    void SerializeResults(const std::vector<KernelResult>& results, const std::string& filePath) override;
};

} // namespace ktt
