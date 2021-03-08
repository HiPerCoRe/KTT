#pragma once

#include <istream>
#include <utility>
#include <vector>

#include <Api/Output/KernelResult.h>
#include <Output/TunerMetadata.h>

namespace ktt
{

class Deserializer
{
public:
    virtual ~Deserializer() = default;

    virtual std::pair<TunerMetadata, std::vector<KernelResult>> DeserializeResults(std::istream& source) = 0;
};

} // namespace ktt
