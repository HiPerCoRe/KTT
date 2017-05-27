#pragma once

#if defined(_WIN32) && !defined(KTT_TESTS)
    #pragma warning(disable : 4251)
    #if defined(KTT_LIBRARY)
        #define KTT_API __declspec(dllexport)
    #else
        #define KTT_API __declspec(dllimport)
    #endif // KTT_LIBRARY
#else
    #define KTT_API
#endif // _WIN32

#include <utility>

#include "../ktt_type_aliases.h"
#include "../enum/thread_size_usage.h"
#include "../tuning_runner/manipulator_interface.h"

namespace ktt
{

class TuningRunner;

class KTT_API TuningManipulator
{
public:
    // Virtual methods
    virtual ~TuningManipulator();
    virtual void launchComputation(const size_t kernelId) = 0;
    virtual std::vector<std::pair<size_t, ThreadSizeUsage>> getUtilizedKernelIds() const;

    // Kernel run methods
    std::vector<ResultArgument> runKernel(const size_t kernelId);
    std::vector<ResultArgument> runKernel(const size_t kernelId, const DimensionVector& globalSize, const DimensionVector& localSize);

    // Configuration retrieval methods
    DimensionVector getCurrentGlobalSize(const size_t kernelId) const;
    DimensionVector getCurrentLocalSize(const size_t kernelId) const;
    std::vector<ParameterValue> getCurrentConfiguration() const;

    // Argument update methods
    void updateArgumentScalar(const size_t argumentId, const void* argumentData);
    void updateArgumentVector(const size_t argumentId, const void* argumentData);
    void updateArgumentVector(const size_t argumentId, const void* argumentData, const size_t numberOfElements);
    void setAutomaticArgumentUpdate(const bool flag);
    void setArgumentSynchronization(const bool flag, const ArgumentMemoryType& argumentMemoryType);
    void updateKernelArguments(const size_t kernelId, const std::vector<size_t>& argumentIds);
    void swapKernelArguments(const size_t kernelId, const size_t argumentIdFirst, const size_t argumentIdSecond);

    // Utility methods
    static std::vector<size_t> convertFromDimensionVector(const DimensionVector& vector);
    static DimensionVector convertToDimensionVector(const std::vector<size_t>& vector);

    friend class TuningRunner;

private:
    ManipulatorInterface* manipulatorInterface;
};

} // namespace ktt
