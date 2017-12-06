#pragma once

#include <map>
#include <string>
#include <vector>
#include "dto/kernel_result.h"
#include "enum/time_unit.h"

namespace ktt
{

class ResultLoader
{
public:
    ResultLoader();
    bool loadResults(const std::string& filePath);

    KernelResult readResult(const KernelConfiguration& configuration);

private:
    std::vector<std::vector<int>> values;
    int timeIndex;
    int paramsBegin;
    int paramsLength;
};

} // namespace ktt
