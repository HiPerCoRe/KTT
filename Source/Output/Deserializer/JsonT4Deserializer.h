#pragma once

#include <Output/Deserializer/Deserializer.h>

namespace ktt
{

class JsonT4Deserializer : public Deserializer
{
public:
    std::pair<TunerMetadata, std::vector<KernelResult>> DeserializeResults(UserData& data, std::istream& source) override;
};

} // namespace ktt
