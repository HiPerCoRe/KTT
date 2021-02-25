#pragma once

#include <istream>
#include <vector>

#include <Api/Output/KernelResult.h>

namespace ktt
{

class Deserializer
{
public:
    virtual ~Deserializer() = default;

    virtual std::vector<KernelResult> DeserializeResults(std::istream& source) = 0;
};

} // namespace ktt
