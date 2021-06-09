#pragma once

#include <string>
#include <vector>

#include <Api/Configuration/DimensionVector.h>
#include <KernelArgument/KernelArgument.h>
#include <KttTypes.h>

namespace ktt
{

class Kernel;
class KernelConfiguration;
class KernelDefinition;

class KernelComputeData
{
public:
    explicit KernelComputeData(const Kernel& kernel, const KernelDefinition& definition, const KernelConfiguration& configuration);

    void SetGlobalSize(const DimensionVector& globalSize);
    void SetLocalSize(const DimensionVector& localSize);
    void UpdateArgumentAtIndex(const size_t index, KernelArgument& argument);
    void SwapArguments(const ArgumentId first, const ArgumentId second);
    void SetArguments(const std::vector<KernelArgument*> arguments);

    const std::string& GetName() const;
    const std::string& GetDefaultSource() const;
    const std::string& GetConfigurationPrefix() const;
    std::string GetSource() const;
    const std::string& GetTemplatedName() const;
    KernelComputeId GetUniqueIdentifier() const;
    const DimensionVector& GetGlobalSize() const;
    const DimensionVector& GetLocalSize() const;
    const KernelConfiguration& GetConfiguration() const;
    size_t GetArgumentIndex(const ArgumentId id) const;
    const std::vector<KernelArgument*>& GetArguments() const;

private:
    std::string m_Name;
    std::string m_DefaultSource;
    std::string m_ConfigurationPrefix;
    std::string m_TemplatedName;
    DimensionVector m_GlobalSize;
    DimensionVector m_LocalSize;
    const KernelConfiguration* m_Configuration;
    std::vector<KernelArgument*> m_Arguments;
};

} // namespace ktt
