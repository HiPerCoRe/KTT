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
        const DimensionVector& globalSize, const DimensionVector& localSize);

    void SetArguments(const std::vector<KernelArgument*> arguments);

    KernelDefinitionId GetId() const;
    const std::string& GetName() const;
    const std::string& GetSource() const;
    const DimensionVector& GetGlobalSize() const;
    const DimensionVector& GetLocalSize() const;
    const std::vector<KernelArgument*>& GetArguments() const;
    std::vector<KernelArgument*> GetVectorArguments() const;
    bool HasArgument(const ArgumentId id) const;

private:
    KernelDefinitionId m_Id;
    std::string m_Name;
    std::string m_Source;
    DimensionVector m_GlobalSize;
    DimensionVector m_LocalSize;
    std::vector<KernelArgument*> m_Arguments;
};

} // namespace ktt
