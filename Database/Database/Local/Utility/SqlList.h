#pragma once

#include <cstddef>
#include <string>

namespace ktt::db
{

/** @fn std::string SqlList(const size_t size)
 * Builds a parenthesized list of SQL parameter placeholders, e.g. "(?, ?, ?)" for size 3.
 * @param size Number of placeholders.
 * @return The placeholder list.
 */
std::string SqlList(const size_t size);

} // namespace ktt::db
