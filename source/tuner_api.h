#pragma once

#include <memory>
#include <string>
#include <vector>

// Type aliases and enums relevant to usage of API methods
#include "ktt_type_aliases.h"
#include "enums/dimension.h"
#include "enums/argument_memory_type.h"
#include "enums/search_method.h"
#include "enums/thread_modifier_type.h"

// Information about platforms and devices
#include "dtos/device_info.h"
#include "dtos/platform_info.h"

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
    void addParameter(const size_t kernelId, const std::string& name, const std::vector<size_t>& values);
    void addParameter(const size_t kernelId, const std::string& name, const std::vector<size_t>& values,
        const ThreadModifierType& threadModifierType, const Dimension& modifierDimension);
    void setKernelArguments(const size_t kernelId, const std::vector<size_t>& argumentIds);
    void setSearchMethod(const size_t kernelId, const SearchMethod& searchMethod, const std::vector<double>& searchArguments);

    // Argument handling methods
    template <typename T> size_t addArgument(const std::vector<T>& data, const ArgumentMemoryType& argumentMemoryType);
    template <typename T> size_t addArgument(const T value);
    template <typename T> void updateArgument(const size_t argumentId, const std::vector<T>& data);
    template <typename T> void updateArgument(const size_t argumentId, const T value);

    // Kernel tuning methods
    void tuneKernel(const size_t kernelId);

    // Compute API methods
    static void printComputeAPIInfo(std::ostream& outputTarget);
    static PlatformInfo getPlatformInfo(const size_t platformIndex);
    static std::vector<PlatformInfo> getPlatformInfoAll();
    static DeviceInfo getDeviceInfo(const size_t platformIndex, const size_t deviceIndex);
    static std::vector<DeviceInfo> getDeviceInfoAll(const size_t platformIndex);
    void setCompilerOptions(const std::string& options);

private:
    // Attributes
    std::unique_ptr<TunerCore> tunerCore;
};

} // namespace ktt
