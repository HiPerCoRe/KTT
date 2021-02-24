#pragma once

#include <Output/Serializer/Serializer.h>

namespace ktt
{

class JsonSerializer : public Serializer
{
public:
    void SerializeResults(const TunerMetadata& metadata, const std::vector<KernelResult>& results, std::ostream& target) override;
};

} // namespace ktt
