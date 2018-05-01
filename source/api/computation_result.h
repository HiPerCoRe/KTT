/** @file computation_result.h
  * Retrieval of information about computation using specific configuration.
  */
#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "ktt_platform.h"
#include "api/parameter_pair.h"

namespace ktt
{

/** @class ComputationResult
  * Class which holds information about computation in specific configuration.
  */
class KTT_API ComputationResult
{
public:
    /** @fn ComputationResult()
      * Default constructor, initializes status to false and duration to maximum value.
      */
    ComputationResult();

    /** @fn explicit ComputationResult(const std::string& kernelName, const std::vector<ParameterPair>& configuration, const uint64_t duration)
      * Constructor which initializes tuning result for successful computations. Error message is initialized to empty string.
      * @param kernelName Name of a kernel.
      * @param configuration Configuration used for computation.
      * @param duration Duration of a computation using specified configuration.
      */
    explicit ComputationResult(const std::string& kernelName, const std::vector<ParameterPair>& configuration, const uint64_t duration);

    /** @fn explicit ComputationResult(const std::string& kernelName, const std::vector<ParameterPair>& configuration,
      * const std::string& errorMessage)
      * Constructor which initializes tuning result for failed computations. Duration is initialized to maximum value.
      * @param kernelName Name of a kernel.
      * @param configuration Configuration used for computation.
      * @param errorMessage Contains information about computation failure.
      */
    explicit ComputationResult(const std::string& kernelName, const std::vector<ParameterPair>& configuration, const std::string& errorMessage);

    /** @fn bool getStatus() const
      * Getter for computation status.
      * @return True if computation was successful. False otherwise.
      */
    bool getStatus() const;

    /** @fn uint64_t getDuration() const
      * Getter for computation duration.
      * @return If status is true, actual duration of computation in nanoseconds. If status is false, maximum value.
      */
    uint64_t getDuration() const;

    /** @fn std::string getKernelName() const
      * Getter for kernel name.
      * @return Kernel name.
      */
    std::string getKernelName() const;

    /** @fn std::string getErrorMessage() const
      * Getter for error message.
      * @return If status is true, empty string. If status is false, message containing information about computation failure.
      */
    std::string getErrorMessage() const;

    /** @fn std::vector<ParameterPair> getConfiguration() const
      * Getter for tuning result configuration.
      * @return Tuning result configuration.
      */
    std::vector<ParameterPair> getConfiguration() const;

private:
    bool status;
    uint64_t duration;
    std::string kernelName;
    std::string errorMessage;
    std::vector<ParameterPair> configuration;
};

} // namespace ktt
