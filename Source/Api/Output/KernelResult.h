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
      * Sets duration of a kernel launcher. The duration of buffer transfers performed within the launcher is not included.
      * @param duration Kernel launcher duration.
      */
    void SetExtraDuration(const Nanoseconds duration);

    /** @fn void SetExtraOverhead(const Nanoseconds overhead)
      * Sets duration of buffer transfers (e.g., between host and device memory).
      * @param overhead Duration of buffer transfers.
      */
    [[deprecated("Use SetDataMovementOverhead() instead.")]] void SetExtraOverhead(const Nanoseconds overhead);

    /** @fn void SetDataMovementOverhead(const Nanoseconds overhead)
      * Sets duration of buffer transfers (e.g., between host and device memory).
      * @param overhead Duration of buffer transfers.
      */
    void SetDataMovementOverhead(const Nanoseconds overhead);

    /** @fn void SetValidationOverhead(const Nanoseconds overhead)
      * Sets duration of kernel output validation.
      * @param overhead Duration of kernel output validation.
      */
    void SetValidationOverhead(const Nanoseconds overhead);

    /** @fn void SetSearcherOverhead(const Nanoseconds overhead)
      * Sets duration of searcher finding the next configuration to run.
      * @param overhead Duration of searcher finding the next configuration to run.
      */
    void SetSearcherOverhead(const Nanoseconds overhead);

    /** @fn void SetFailedKernelOverhead(const Nanoseconds overhead)
      * Sets duration of failed (uncompilable) kernel creation.
      * @param overhead Duration of KTT/compiler spent on preparation of kernel which cannot be executed.
      */
    void SetFailedKernelOverhead(const Nanoseconds overhead);

    /** @fn void SetProfilingRunsOverhead(const Nanoseconds overhead)
      * Sets time of kernels executed to collect performance counters.
      * @param Duration of kernels executed just to collect performance counters.
      */
    void SetProfilingRunsOverhead(const Nanoseconds overhead);

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

    /** @fn Nanoseconds GetKernelCompilationOverhead() const
      * Retrieves total kernel compilation overhead from the partial results.
      * @return Total kernel compilation overhead from the partial results.
      */
    Nanoseconds GetKernelCompilationOverhead() const;

    /** @fn Nanoseconds GetExtraDuration() const
      * Retrieves duration of a kernel launcher. The duration of buffer transfers performed within the launcher is not included.
      * @return Kernel launcher duration.
      */
    Nanoseconds GetExtraDuration() const;

    /** @fn Nanoseconds GetExtraOverhead() const
      * Retrieves duration of buffer transfers (e.g., between host and device memory).
      * @return Duration of buffer transfers.
      */
    [[deprecated("Use GetDataMovementOverhead() instead.")]] Nanoseconds GetExtraOverhead() const;

    /** @fn Nanoseconds GetDataMovementOverhead() const
      * Retrieves duration of buffer transfers (e.g., between host and device memory).
      * @return Duration of buffer transfers.
      */
    Nanoseconds GetDataMovementOverhead() const;

    /** @fn Nanoseconds GetValidationOverhead() const
      * Retrieves duration of kernel output validation.
      * @return Duration of kernel output validation.
      */
    Nanoseconds GetValidationOverhead() const;

    /** @fn Nanoseconds GetSearcherOverhead() const
      * Retrieves duration of searcher finding the next configuration to run.
      * @return Duration of searcher finding the next configuration to run.
      */
    Nanoseconds GetSearcherOverhead() const;

    /** @fn Nanoseconds GetFailedKernelOverhead() const
      * Retrieves duration of failed (uncompilable) kernel creation.
      * @return Duration of KTT/compiler spent on preparation of kernel which cannot be executed.
      */
    Nanoseconds GetFailedKernelOverhead() const;

    /** @fn Nanoseconds GetProfilingRunsOverhead() const
      * Retrieves duration and overhead of kernels executed to collect performance counters.
      * @return Duration of kernels executed just to collect performance 
      * counters.
      */
    Nanoseconds GetProfilingRunsOverhead() const;

    /** @fn Nanoseconds GetProfilingOverhead() const
      * Retrieves duration of all non-kernel operations performed during collection performance counters (e.g., data movements for extra kernel runs).
      * @return Duration operations different than kernel execution needed to coollect performance counters.
      */
    Nanoseconds GetProfilingOverhead() const;

    /** @fn Nanoseconds GetProfilingTotalOverhead() const
      * Retrieves duration and overhead of all operations needed to collect performance counters.
      * @return Duration of operations required just to collect performance
      * counters.
      */
    Nanoseconds GetProfilingTotalOverhead() const;

    /** @fn Nanoseconds GetCompilationOverhead() const
      * Retrieves duration of kernels compilation.
      * @return Duration of kernels compilation.
      */
    Nanoseconds GetCompilationOverhead() const;

    /** @fn Nanoseconds GetTotalDuration() const
      * Retrieves the sum of kernel duration and extra duration.
      * @return Sum of kernel duration and extra duration.
      */
    Nanoseconds GetTotalDuration() const;

    /** @fn Nanoseconds GetTotalOverhead() const
      * Retrieves the sum of kernel, data movement, validation and searcher overhead.
      * @return The sum of kernel, data movement, validation and searcher overhead.
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

    /** @fn void FuseProfilingTimes(const KernelResult& previousResult)
     * Fuse overhead times collected from multiple profiling executions
     * @param previousResult result of the previous run
     * @param first sets whether the first instance of the kernel configuration was computed (we need to fuse overhead time, but do not count it as overhead of profiling).
      */
    void FuseProfilingTimes(const KernelResult& previousResult, bool first);

    /** @fn void CopyProfilingTimes(const KernelResult& originalResult)
     * Copy overhead times collected from another KernelResult instance
     * @param originalResult source of overhead times
      */
    void CopyProfilingTimes(const KernelResult& originalResult);

private:
    KernelConfiguration m_Configuration;
    std::vector<ComputationResult> m_Results;
    std::string m_KernelName;
    Nanoseconds m_ExtraDuration;
    Nanoseconds m_DataMovementOverhead;
    Nanoseconds m_ValidationOverhead;
    Nanoseconds m_SearcherOverhead;
    Nanoseconds m_FailedKernelOverhead;
    Nanoseconds m_ProfilingRunsOverhead;
    Nanoseconds m_ProfilingOverhead;
    Nanoseconds m_CompilationOverhead;
    ResultStatus m_Status;
};

} // namespace ktt
