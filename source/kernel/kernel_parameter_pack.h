#pragma once

#include <string>
#include <utility>
#include <vector>
#include "ktt_types.h"

namespace ktt
{

class KernelParameterPack
{
public:
    KernelParameterPack();
    explicit KernelParameterPack(const std::string& name, const std::vector<std::string>& parameterNames);

    std::string getName() const;
    std::vector<std::string> getParameterNames() const;

    bool operator==(const KernelParameterPack& other) const;
    bool operator!=(const KernelParameterPack& other) const;

private:
    std::string name;
    std::vector<std::string> parameterNames;
};

} // namespace ktt
