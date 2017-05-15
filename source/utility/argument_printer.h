#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "logger.h"
#include "../enum/argument_print_condition.h"
#include "../kernel/kernel_configuration.h"
#include "../kernel_argument/kernel_argument.h"

namespace ktt
{

class ArgumentPrinter
{
public:
    explicit ArgumentPrinter(Logger* logger);

    void printArgument(const KernelArgument& kernelArgument, const std::string& kernelName, const KernelConfiguration& kernelConfiguration,
        const bool resultIsValid) const;
    void setArgumentPrintData(const size_t argumentId, const std::string& filePath, const ArgumentPrintCondition& argumentPrintCondition);

    bool argumentPrintDataExists(const size_t argumentId) const;

private:
    std::map<size_t, std::pair<std::string, ArgumentPrintCondition>> argumentPrintData;
    Logger* logger;
};

} // namespace ktt
