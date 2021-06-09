#pragma once

#include <Output/Deserializer/Deserializer.h>

namespace ktt
{

class JsonDeserializer : public Deserializer
{
public:
    std::pair<TunerMetadata, std::vector<KernelResult>> DeserializeResults(UserData& data, std::istream& source) override;
};

} // namespace ktt
