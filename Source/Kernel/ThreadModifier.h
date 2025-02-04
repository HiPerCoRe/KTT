#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <Api/Configuration/ParameterPair.h>
#include <KttTypes.h>

namespace ktt
{

class ThreadModifier
{
public:
    ThreadModifier();
    explicit ThreadModifier(const std::vector<std::string>& parameters, const std::vector<KernelDefinitionId>& definitions,
        ModifierFunction function);
    explicit ThreadModifier(const std::vector<KernelDefinitionId>& definitions, const std::string& script);

    const std::vector<std::string>& GetParameters() const;
    const std::vector<KernelDefinitionId>& GetDefinitions() const;
    uint64_t GetModifiedSize(const KernelDefinitionId id, const uint64_t originalSize,
        const std::vector<ParameterPair>& pairs) const;

private:
    std::vector<std::string> m_Parameters;
    std::vector<KernelDefinitionId> m_Definitions;
    ModifierFunction m_Function;
    std::string m_Script;

    uint64_t EvaluateScript(const uint64_t originalSize, const std::vector<ParameterPair>& pairs) const;
};

} // namespace ktt
