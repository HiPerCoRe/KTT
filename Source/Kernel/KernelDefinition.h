#pragma once

#include <string>
#include <vector>

#include <Api/Configuration/DimensionVector.h>
#include <KernelArgument/KernelArgument.h>
#include <KttTypes.h>

namespace ktt
{

class KernelDefinition
{
public:
    explicit KernelDefinition(const KernelDefinitionId id, const std::string& name, const std::string& source,
        const DimensionVector& globalSize, const DimensionVector& localSize, const std::vector<std::string>& typeNames = {});

    void SetArguments(const std::vector<KernelArgument*> arguments);

    KernelDefinitionId GetId() const;
    const std::string& GetName() const;
    const std::string& GetSource() const;
    const std::string& GetTemplatedName() const;
    const DimensionVector& GetGlobalSize() const;
    const DimensionVector& GetLocalSize() const;
    const std::vector<KernelArgument*>& GetArguments() const;
    std::vector<KernelArgument*> GetVectorArguments() const;
    bool HasArgument(const ArgumentId id) const;

    static std::string CreateTemplatedName(const std::string& name, const std::vector<std::string>& typeNames);

private:
    KernelDefinitionId m_Id;
    std::string m_Name;
    std::string m_Source;
    std::string m_TemplatedName;
    DimensionVector m_GlobalSize;
    DimensionVector m_LocalSize;
    std::vector<KernelArgument*> m_Arguments;
};

} // namespace ktt
