#include <cstddef>
#include <string>

#include <Api/Output/KernelResult.h>
#include <Output/OutputFormat.h>

namespace ktt::db
{
/** @class DatabaseUtility
 * Utility functions for building SQL and (de)serializing stored kernel results.
 * Statement preparation, binding and reading is handled by Statement.
 */
class DatabaseUtility
{
public:
    /** @fn static std::string SqlList(const size_t size)
     * Builds a parenthesized list of SQL parameter placeholders, e.g. "(?, ?, ?)" for size 3.
     * @param size Number of placeholders.
     * @return The placeholder list.
     */
    static std::string SqlList(const size_t size);

    /** @fn static std::string SerializeResult(const KernelResult& result, ktt::OutputFormat format, int indent)
     * Serializes a single kernel result into the textual representation of the given output format.
     * @param result Kernel result to serialize.
     * @param format Output format to serialize into.
     * @param indent Indentation level applied to JSON output (ignored for XML).
     * @return Serialized result stored in the tuning_result.result column.
     */
    static std::string SerializeResult(const KernelResult& result, ktt::OutputFormat format, int indent);

    /** @fn static KernelResult DeserializeResult(const std::string& text, ktt::OutputFormat format)
     * Parses a stored result string back into a KernelResult using the format it was serialized with.
     * @param text Serialized result payload.
     * @param format Output format the result was serialized with.
     * @return Reconstructed kernel result.
     */
    static KernelResult DeserializeResult(const std::string& text, ktt::OutputFormat format);
};

} // namespace ktt::db
