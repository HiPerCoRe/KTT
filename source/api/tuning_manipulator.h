#pragma once

#ifndef KTT_API
#if defined(_MSC_VER) && !defined(KTT_TESTS)
    #pragma warning(disable : 4251) // MSVC irrelevant warning (as long as there are no public attributes)
    #if defined(KTT_LIBRARY)
        #define KTT_API __declspec(dllexport)
    #else
        #define KTT_API __declspec(dllimport)
    #endif // KTT_LIBRARY
#else
    #define KTT_API
#endif // _MSC_VER
#endif // KTT_API

#include <cstddef>
#include <utility>
#include <vector>

#include "ktt_type_aliases.h"
#include "enum/argument_data_type.h"
#include "enum/argument_location.h"
#include "enum/argument_memory_type.h"
#include "enum/thread_size_usage.h"

namespace ktt
{

class TuningRunner;
class ManipulatorInterface;

class KTT_API TuningManipulator
{
public:
    // Virtual methods
    virtual ~TuningManipulator();
    virtual void launchComputation(const size_t kernelId) = 0;
    virtual std::vector<std::pair<size_t, ThreadSizeUsage>> getUtilizedKernelIds() const;

    // Kernel run methods
    void runKernel(const size_t kernelId);
    void runKernel(const size_t kernelId, const DimensionVector& globalSize, const DimensionVector& localSize);

    // Configuration retrieval methods
    DimensionVector getCurrentGlobalSize(const size_t kernelId) const;
    DimensionVector getCurrentLocalSize(const size_t kernelId) const;
    std::vector<ParameterValue> getCurrentConfiguration() const;

    // Argument update and synchronization methods
    void updateArgumentScalar(const size_t argumentId, const void* argumentData);
    void updateArgumentVector(const size_t argumentId, const void* argumentData, const ArgumentLocation& argumentLocation);
    void updateArgumentVector(const size_t argumentId, const void* argumentData, const ArgumentLocation& argumentLocation,
        const size_t numberOfElements);
    void synchronizeArgumentVector(const size_t argumentId, const bool downloadToHost);

    // Kernel argument handling methods
    void setKernelArguments(const size_t kernelId, const std::vector<size_t>& argumentIds);
    void swapKernelArguments(const size_t kernelId, const size_t argumentIdFirst, const size_t argumentIdSecond);

    // Utility methods
    static std::vector<size_t> convertFromDimensionVector(const DimensionVector& vector);
    static DimensionVector convertToDimensionVector(const std::vector<size_t>& vector);
    static size_t getParameterValue(const std::string& parameterName, const std::vector<ParameterValue>& parameterValues);

    friend class TuningRunner;

private:
    ManipulatorInterface* manipulatorInterface;
};

} // namespace ktt
