#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <Api/ParameterPair.h>
#include <KttTypes.h>

namespace ktt
{

class ThreadModifier
{
public:
    ThreadModifier();
    explicit ThreadModifier(const std::vector<std::string>& parameters, const std::vector<KernelDefinitionId>& definitions,
        std::function<uint64_t(const uint64_t, const std::vector<uint64_t>&)> function);

    const std::vector<std::string>& GetParameters() const;
    const std::vector<KernelDefinitionId>& GetDefinitions() const;
    uint64_t GetModifiedSize(const KernelDefinitionId id, const uint64_t originalSize,
        const std::vector<ParameterPair>& pairs) const;

private:
    std::vector<std::string> m_Parameters;
    std::vector<KernelDefinitionId> m_Definitions;
    std::function<uint64_t(const uint64_t, const std::vector<uint64_t>&)> m_Function;
};

} // namespace ktt
