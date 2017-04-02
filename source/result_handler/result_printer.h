#pragma once

#include <vector>

#include "../dto/tuning_result.h"

namespace ktt
{

class ResultPrinter
{
public:
    ResultPrinter() = default;

    std::vector<TuningResult> getResults() const
    {
        return results;
    }

    void setResults(const std::vector<TuningResult>& results)
    {
        this->results = results;
    }

private:
    std::vector<TuningResult> results;
};

} // namespace ktt
