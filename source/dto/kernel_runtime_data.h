#pragma once

#include <string>
#include <vector>

#include "ktt_type_aliases.h"
#include "kernel_argument/kernel_argument.h"

namespace ktt
{

class KernelRuntimeData
{
public:
    KernelRuntimeData(const std::string& name, const std::string& source, const DimensionVector& globalSize, const DimensionVector& localSize,
        const std::vector<size_t>& argumentIndices);

    void setArgumentIndices(const std::vector<size_t>& argumentIndices);

    std::string getName() const;
    std::string getSource() const;
    DimensionVector getGlobalSize() const;
    DimensionVector getLocalSize() const;
    std::vector<size_t> getArgumentIndices() const;

private:
    std::string name;
    std::string source;
    DimensionVector globalSize;
    DimensionVector localSize;
    std::vector<size_t> argumentIndices;
};

} // namespace ktt
