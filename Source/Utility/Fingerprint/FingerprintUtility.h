#pragma once
#include <cstddef>
#include <set>
#include <string_view>
#include <vector>

#include <Kernel/KernelConstraint/KernelConstraint.h>
#include <Kernel/KernelDefinition.h>
#include <Kernel/KernelParameter.h>

namespace ktt
{

class FingerprintUtility
{
public:
    static std::size_t GetFingerprintOfParameters(const std::set<KernelParameter> &params);
    static std::size_t GetFingerprintOfDefinitions(const std::vector<const KernelDefinition *> &definitions);

    static std::size_t HashString(std::string_view value);
    static std::size_t HashFunction(std::size_t base, std::size_t value);
};

} // namespace ktt
