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
    explicit KernelParameter(const std::string& name, const std::vector<uint64_t>& values, const std::string& group);
    explicit KernelParameter(const std::string& name, const std::vector<double>& values, const std::string& group);

    const std::string& GetName() const;
    const std::string& GetGroup() const;
    size_t GetValuesCount() const;
    const std::vector<uint64_t>& GetValues() const;
    const std::vector<double>& GetValuesDouble() const;
    bool HasValuesDouble() const;
    ParameterPair GeneratePair(const size_t valueIndex) const;
    std::vector<ParameterPair> GeneratePairs() const;

    bool operator==(const KernelParameter& other) const;
    bool operator!=(const KernelParameter& other) const;
    bool operator<(const KernelParameter& other) const;

private:
    std::string m_Name;
    std::string m_Group;
    std::variant<std::vector<uint64_t>, std::vector<double>> m_Values;
};

} // namespace ktt
