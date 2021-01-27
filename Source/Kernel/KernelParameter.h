#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include <KttTypes.h>

namespace ktt
{

class KernelParameter
{
public:
    explicit KernelParameter(const std::string& name, const std::vector<uint64_t>& values);
    explicit KernelParameter(const std::string& name, const std::vector<double>& values);

    const std::string& GetName() const;
    size_t GetValuesCount() const;
    const std::vector<uint64_t>& GetValues() const;
    const std::vector<double>& GetValuesDouble() const;
    bool HasValuesDouble() const;

    bool operator==(const KernelParameter& other) const;
    bool operator!=(const KernelParameter& other) const;

private:
    std::string m_Name;
    std::variant<std::vector<uint64_t>, std::vector<double>> m_Values;
};

} // namespace ktt
