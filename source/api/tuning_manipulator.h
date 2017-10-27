#pragma once

#include <cstddef>
#include <utility>
#include <vector>
#include "ktt_platform.h"
#include "ktt_types.h"
#include "api/dimension_vector.h"

namespace ktt
{

class TuningRunner;
class ManipulatorInterface;

class KTT_API TuningManipulator
{
public:
    // Virtual methods
    virtual ~TuningManipulator();
    virtual void launchComputation(const KernelId id) = 0;
    virtual TunerFlag enableArgumentPreload() const;

    // Kernel run methods
    void runKernel(const KernelId id);
    void runKernel(const KernelId id, const DimensionVector& globalSize, const DimensionVector& localSize);

    // Configuration retrieval methods
    DimensionVector getCurrentGlobalSize(const KernelId id) const;
    DimensionVector getCurrentLocalSize(const KernelId id) const;
    std::vector<ParameterPair> getCurrentConfiguration() const;

    // Argument update and retrieval methods
    void updateArgumentScalar(const ArgumentId id, const void* argumentData);
    void updateArgumentLocal(const ArgumentId id, const size_t numberOfElements);
    void updateArgumentVector(const ArgumentId id, const void* argumentData);
    void updateArgumentVector(const ArgumentId id, const void* argumentData, const size_t numberOfElements);
    void getArgumentVector(const ArgumentId id, void* destination) const;
    void getArgumentVector(const ArgumentId id, void* destination, const size_t numberOfElements) const;

    // Kernel argument handling methods
    void changeKernelArguments(const KernelId id, const std::vector<ArgumentId>& argumentIds);
    void swapKernelArguments(const KernelId id, const ArgumentId argumentIdFirst, const ArgumentId argumentIdSecond);

    // Buffer handling methods
    void createArgumentBuffer(const ArgumentId id);
    void destroyArgumentBuffer(const ArgumentId id);

    // Utility methods
    static size_t getParameterValue(const std::string& parameterName, const std::vector<ParameterPair>& parameterPairs);

    friend class TuningRunner;

private:
    ManipulatorInterface* manipulatorInterface;
};

} // namespace ktt
