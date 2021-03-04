/** @file ComputationResult.h
  * Aggregate result data from kernel computation.
  */
#pragma once

#include <vector>

#include <Api/Configuration/KernelConfiguration.h>
#include <Api/Output/ComputationResult.h>
#include <Api/Output/ResultStatus.h>
#include <KttPlatform.h>
#include <KttTypes.h>

namespace ktt
{

/** @class KernelResult
  * Class which holds aggregate result data from kernel computation such as individual computation results, configuration and status.
  */
class KTT_API KernelResult
{
public:
    /** @fn KernelResult()
      * Constructor which creates empty invalid kernel result.
      */
    KernelResult();

    /** @fn explicit KernelResult(const std::string& kernelName, const KernelConfiguration& configuration)
      * Constructor which creates kernel result for the specified kernel and configuration.
      * @param kernelName Name of a kernel tied to the result.
      * @param configuration Configuration tied to the result.
      */
    explicit KernelResult(const std::string& kernelName, const KernelConfiguration& configuration);

    /** @fn explicit KernelResult(const std::string& kernelName, const KernelConfiguration& configuration,
      * const std::vector<ComputationResult>& results)
      * Constructor which creates kernel result for the specified kernel and configuration. Also fills partial results.
      * @param kernelName Name of a kernel tied to the result.
      * @param configuration Configuration tied to the result.
      * @param results Partial results from all kernel definition runs under the kernel launch.
      */
    explicit KernelResult(const std::string& kernelName, const KernelConfiguration& configuration,
        const std::vector<ComputationResult>& results);

    /** @fn void SetStatus(const ResultStatus status)
      * Sets kernel results status.
      * @param status Result status. See ::ResultStatus for more information.
      */
    void SetStatus(const ResultStatus status);

    /** @fn void SetExtraDuration(const Nanoseconds duration)
      * Sets extra duration. This includes for example duration of custom kernel launcher minus the duration of buffer transfers.
      * @param duration Extra computation duration.
      */
    void SetExtraDuration(const Nanoseconds duration);

    /** @fn void SetExtraOverhead(const Nanoseconds overhead)
      * Sets extra overhead. This includes for example duration of buffer transfers performed in custom kernel launcher.
      * @param overhead Extra computation overhead.
      */
    void SetExtraOverhead(const Nanoseconds overhead);

    /** @fn const std::string& GetKernelName() const
      * Returns name of a kernel tied to the result.
      * @return Name of a kernel tied to the result.
      */
    const std::string& GetKernelName() const;

    /** @fn const std::vector<ComputationResult>& GetResults() const
      * Retrieves partial results from computations performed as part of the kernel launch.
      * @return Partial results from computations. See ComputationResult for more information.
      */
    const std::vector<ComputationResult>& GetResults() const;

    /** @fn const KernelConfiguration& GetConfiguration() const
      * Retrieves kernel configuration tied to the result.
      * @return Kernel configuration tied to the result.
      */
    const KernelConfiguration& GetConfiguration() const;

    /** @fn ResultStatus GetStatus() const
      * Retrieves status of the kernel result.
      * @return Status of the kernel result. See ::ResultStatus for more information.
      */
    ResultStatus GetStatus() const;

    /** @fn Nanoseconds GetKernelDuration() const
      * Retrieves total kernel duration from the partial results.
      * @return Total kernel duration from the partial results.
      */
    Nanoseconds GetKernelDuration() const;

    /** @fn Nanoseconds GetKernelOverhead() const
      * Retrieves total kernel overhead from the partial results.
      * @return Total kernel overhead from the partial results.
      */
    Nanoseconds GetKernelOverhead() const;

    /** @fn Nanoseconds GetExtraDuration() const
      * Retrieves extra kernel result duration. This includes for example duration of custom kernel launcher minus the duration
      * of buffer transfers.
      * @return Extra kernel result duration.
      */
    Nanoseconds GetExtraDuration() const;

    /** @fn Nanoseconds GetExtraOverhead() const
      * Retrieves extra kernel result overhead. This includes for example duration of buffer transfers performed in custom kernel
      * launcher.
      * @return Extra kernel result overhead.
      */
    Nanoseconds GetExtraOverhead() const;

    /** @fn Nanoseconds GetTotalDuration() const
      * Retrieves the sum of kernel duration and extra duration.
      * @return Sum of kernel duration and extra duration.
      */
    Nanoseconds GetTotalDuration() const;

    /** @fn Nanoseconds GetTotalOverhead() const
      * Retrieves the sum of kernel overhead and extra overhead.
      * @return Sum of kernel overhead and extra overhead.
      */
    Nanoseconds GetTotalOverhead() const;

    /** @fn bool IsValid() const
      * Checks whether kernel result is valid. I.e., its status has value Ok.
      * @return True if kernel result is valid. False otherwise.
      */
    bool IsValid() const;

    /** @fn bool HasRemainingProfilingRuns() const
      * Checks whether more kernel runs under the corresponding configuration need to be performed before all of the partial
      * results have profiling data with valid information.
      * @return True if more kernel runs need to be performed under the same configuration. False otherwise.
      */
    bool HasRemainingProfilingRuns() const;

private:
    KernelConfiguration m_Configuration;
    std::vector<ComputationResult> m_Results;
    std::string m_KernelName;
    Nanoseconds m_ExtraDuration;
    Nanoseconds m_ExtraOverhead;
    ResultStatus m_Status;
};

} // namespace ktt
