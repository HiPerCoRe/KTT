#pragma once

#include <memory>
#include <string>
#include <vector>

// Type aliases and enums relevant to usage of API methods
#include "ktt_type_aliases.h"
#include "enums/dimension.h"
#include "enums/kernel_argument_access_type.h"
#include "enums/search_method.h"
#include "enums/thread_modifier_type.h"

namespace ktt
{

class TunerCore; // Forward declaration of TunerCore class

class Tuner
{
public:
    // Constructor and destructor
    explicit Tuner(const size_t platformIndex, const size_t deviceIndex);
    ~Tuner();

    // Kernel handling methods
    size_t addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize, const DimensionVector& localSize);
    size_t addKernelFromFile(const std::string& filePath, const std::string& kernelName, const DimensionVector& globalSize,
        const DimensionVector& localSize);
    void addParameter(const size_t id, const std::string& name, const std::vector<size_t>& values);
    void addParameter(const size_t id, const std::string& name, const std::vector<size_t>& values, const ThreadModifierType& threadModifierType,
        const Dimension& modifierDimension);
    void addArgumentInt(const size_t id, const std::vector<int>& data, const KernelArgumentAccessType& kernelArgumentAccessType);
    void addArgumentFloat(const size_t id, const std::vector<float>& data, const KernelArgumentAccessType& kernelArgumentAccessType);
    void addArgumentDouble(const size_t id, const std::vector<double>& data, const KernelArgumentAccessType& kernelArgumentAccessType);
    void useSearchMethod(const size_t id, const SearchMethod& searchMethod, const std::vector<double>& searchArguments);

    // Compute API methods
    static void printComputeAPIInfo(std::ostream& outputTarget);
    void setCompilerOptions(const std::string& options);

private:
    // Attributes
    std::unique_ptr<TunerCore> tunerCore;
};

} // namespace ktt
