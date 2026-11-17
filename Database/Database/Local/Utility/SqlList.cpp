#include <Database/Local/Utility/SqlList.h>

namespace ktt::db
{

std::string SqlList(const size_t size)
{
    std::string sqlList = "(";
    for (size_t i = 0; i < size; ++i)
    {
        if (i > 0)
            sqlList += ", ";
        sqlList += "?";
    }
    sqlList += ")";
    return sqlList;
}

} // namespace ktt::db
