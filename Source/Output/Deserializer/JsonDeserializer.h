#pragma once

#include <Output/Deserializer/Deserializer.h>

namespace ktt
{

class JsonDeserializer : public Deserializer
{
public:
    std::vector<KernelResult> DeserializeResults(std::istream& source) override;
};

} // namespace ktt
