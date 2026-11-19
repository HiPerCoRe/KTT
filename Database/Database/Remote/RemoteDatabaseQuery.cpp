#include <cstdint>
#include <string>
#include <vector>

#include <json.hpp>

#include <Api/KttException.h>
#include <Database/Remote/RemoteDatabase.h>
#include <Database/Remote/Utility/RemoteCall.h>
#include <Database/Remote/Utility/RemoteClient.h>
#include <Database/Remote/Utility/Wire/Query.h>

namespace ktt::db
{

namespace
{

using json = nlohmann::json;

// The server's limits for $top and $skip (query.md, "Limits and safety").
constexpr uint64_t MaxTop = 1000;
constexpr uint64_t MaxSkip = 100000;

} // namespace

std::vector<ResultQueryItem> RemoteDatabase::QueryResults(const ResultQuery &query) const
{
    static const std::string Operation = "RemoteDatabase::QueryResults";

    // As $top=0 in OData, and as a limit of 0 in the other reads.
    if (query.top == 0)
        return {};

    if (query.top > MaxTop || query.skip > MaxSkip)
    {
        throw KttException(Operation + ": top must be at most " + std::to_string(MaxTop) + " and skip at most "
            + std::to_string(MaxSkip) + ", got top " + std::to_string(query.top) + " and skip "
            + std::to_string(query.skip));
    }

    std::vector<ResultQueryItem> items;

    TranslateRemote(Operation, *m_Client, [&]
    {
        const json body = wire::ResultQueryRequest::From(query);
        // The endpoint has no "not found": an unknown source or space in the filter matches nothing (200).
        const auto response = CallRemote(*m_Client, Operation, "POST", "/result/query", &body, false);
        const auto wireItems = response->get<wire::ResultQueryResponse>().items;
        items.reserve(wireItems.size());

        for (const auto &item : wireItems)
            items.push_back(item.ToResultQueryItem(query.includeResult));
    });

    return items;
}

} // namespace ktt::db
