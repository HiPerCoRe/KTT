#pragma once

#include <string>
#include <utility>
#include <vector>
#include <ktt_types.h>

namespace ktt
{

class KernelParameter
{
public:
    explicit KernelParameter(const std::string& name, const std::vector<size_t>& values);
    explicit KernelParameter(const std::string& name, const std::vector<double>& values);

    const std::string& getName() const;
    const std::vector<size_t>& getValues() const;
    const std::vector<double>& getValuesDouble() const;
    bool hasValuesDouble() const;

    bool operator==(const KernelParameter& other) const;
    bool operator!=(const KernelParameter& other) const;

private:
    std::string name;
    std::vector<size_t> values;
    std::vector<double> valuesDouble;
    bool isDouble;
};

} // namespace ktt
