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
    std::vector<size_t> getGlobalSize() const;
    std::vector<size_t> getLocalSize() const;
    DimensionVector getGlobalSizeDimensionVector() const;
    DimensionVector getLocalSizeDimensionVector() const;
    std::vector<size_t> getArgumentIndices() const;

private:
    std::string name;
    std::string source;
    std::vector<size_t> globalSize;
    std::vector<size_t> localSize;
    DimensionVector globalSizeDimensionVector;
    DimensionVector localSizeDimensionVector;
    std::vector<size_t> argumentIndices;
};

} // namespace ktt
