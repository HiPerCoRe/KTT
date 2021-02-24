#pragma once

#include <ostream>
#include <vector>

#include <Api/Output/KernelResult.h>
#include <Output/TunerMetadata.h>

namespace ktt
{

class Serializer
{
public:
    virtual ~Serializer() = default;

    virtual void SerializeResults(const TunerMetadata& metadata, const std::vector<KernelResult>& results, std::ostream& target) = 0;
};

} // namespace ktt
