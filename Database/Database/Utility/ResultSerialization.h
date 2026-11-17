#pragma once

#include <string>

#include <Api/Output/KernelResult.h>
#include <Output/OutputFormat.h>
#include <Output/TimeConfiguration/TimeUnit.h>

namespace ktt::db
{
/** @class ResultSerialization
 * (De)serialization of a kernel result to and from the text of an output format. Shared by the local database (the
 * tuning_result.result column) and the remote database (the result sent to and read from ktt-db).
 */
class ResultSerialization
{
public:
    /** @fn static std::string SerializeResult(const KernelResult& result, ktt::OutputFormat format, int indent)
     * Serializes a single kernel result into the textual representation of the given output format.
     * @param result Kernel result to serialize.
     * @param format Output format to serialize into.
     * @param indent Indentation level applied to JSON output (ignored for XML).
     * @return Serialized result.
     */
    static std::string SerializeResult(const KernelResult& result, ktt::OutputFormat format, int indent);

    /** @fn static KernelResult DeserializeResult(const std::string& text, ktt::OutputFormat format, TimeUnit timeUnit)
     * Parses a stored result string back into a KernelResult, reading its durations in the given time unit instead of
     * the time unit that is active now. Use it with the unit the result was serialized with (tuning_run.time_unit), so
     * that changing the unit between saving and loading does not scale the durations. The unit is switched with a
     * TimeUnitGuard for the duration of the call, so this is not thread-safe with respect to other code that reads or
     * sets KTT's time unit.
     * @param text Serialized result payload.
     * @param format Output format the result was serialized with.
     * @param timeUnit Time unit the result's durations were serialized in.
     * @return Reconstructed kernel result, with durations in nanoseconds as usual.
     */
    static KernelResult DeserializeResult(const std::string& text, ktt::OutputFormat format, TimeUnit timeUnit);

private:
    /** @fn static KernelResult DeserializeResult(const std::string& text, ktt::OutputFormat format)
     * Parses a stored result string back into a KernelResult using the format it was serialized with, reading its
     * durations in the time unit that is active now. Private, because that is only right when the active unit is the
     * unit the result was serialized with; callers use the overload that takes the stored time unit.
     * @param text Serialized result payload.
     * @param format Output format the result was serialized with.
     * @return Reconstructed kernel result.
     */
    static KernelResult DeserializeResult(const std::string& text, ktt::OutputFormat format);
};

} // namespace ktt::db
