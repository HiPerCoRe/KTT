#pragma once

#include <memory>
#include <string>

#include "tuner_core.h"

namespace ktt
{

class TunerCore;

class Tuner
{
public:
    // Constructor
    Tuner();

    // Kernel handling methods
    size_t addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize, const DimensionVector& localSize);
    size_t addKernelFromFile(const std::string& filename, const std::string& kernelName, const DimensionVector& globalSize,
        const DimensionVector& localSize);
    void addParameter(const size_t id, const KernelParameter& parameter);
    void addArgumentInt(const size_t id, const std::vector<int>& data);
    void addArgumentFloat(const size_t id, const std::vector<float>& data);
    void addArgumentDouble(const size_t id, const std::vector<double>& data);
    void useSearchMethod(const size_t id, const SearchMethod& searchMethod, const std::vector<double>& searchArguments);

private:
    // Attributes
    std::unique_ptr<TunerCore> tunerCore;
};

} // namespace ktt
