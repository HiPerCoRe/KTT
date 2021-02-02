#pragma once

#include <string>
#include <vector>

#include <Kernel/KernelParameter.h>

namespace ktt
{

class KernelParameterGroup
{
public:
    explicit KernelParameterGroup(const std::string& name, const std::vector<const KernelParameter*>& parameters);

    const std::string& GetName() const;
    const std::vector<const KernelParameter*>& GetParameters() const;
    bool ContainsParameter(const KernelParameter& parameter) const;
    bool ContainsParameter(const std::string& parameter) const;
    uint64_t GetConfigurationsCount() const;

private:
    std::string m_Name;
    std::vector<const KernelParameter*> m_Parameters;
};

} // namespace ktt
