#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include <Api/Configuration/ParameterPair.h>
#include <KttTypes.h>

namespace ktt
{

class KernelParameter
{
public:
    explicit KernelParameter(const std::string& name, const std::vector<ParameterValue>& values, const std::string& group);

    const std::string& GetName() const;
    const std::string& GetGroup() const;
    size_t GetValuesCount() const;
    const std::vector<ParameterValue>& GetValues() const;
    ParameterValueType GetValueType() const;
    ParameterPair GeneratePair(const size_t valueIndex) const;
    std::vector<ParameterPair> GeneratePairs() const;

    bool operator==(const KernelParameter& other) const;
    bool operator!=(const KernelParameter& other) const;
    bool operator<(const KernelParameter& other) const;

private:
    std::string m_Name;
    std::string m_Group;
    std::vector<ParameterValue> m_Values;
};

} // namespace ktt
