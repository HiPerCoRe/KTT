#include <fstream>

#include "argument_printer.h"

namespace ktt
{

ArgumentPrinter::ArgumentPrinter(Logger* logger) :
    logger(logger)
{}

void ArgumentPrinter::printArgument(const KernelArgument& kernelArgument, const std::string& kernelName,
    const KernelConfiguration& kernelConfiguration, const bool resultIsValid) const
{
    if (argumentPrintData.find(kernelArgument.getId()) == argumentPrintData.end())
    {
        throw std::runtime_error(std::string("No argument printing configuration found for argument with id: ")
            + std::to_string(kernelArgument.getId()));
    }

    auto printData = argumentPrintData.find(kernelArgument.getId())->second;
    if (printData.second == ArgumentPrintCondition::ValidOnly && !resultIsValid
        || printData.second == ArgumentPrintCondition::InvalidOnly && resultIsValid)
    {
        return;
    }

    std::ofstream outputFile(printData.first, std::ios::app | std::ios_base::out);

    if (!outputFile.is_open())
    {
        logger->log(std::string("Unable to open file: ") + printData.first);
        return;
    }

    outputFile << "Contents of argument with id " << kernelArgument.getId() << " for kernel with name <" << kernelName
        << "> under configuration: " << kernelConfiguration << "Format is <index: value>" << std::endl;
    outputFile << kernelArgument << std::endl;
}

void ArgumentPrinter::setArgumentPrintData(const size_t argumentId, const std::string& filePath,
    const ArgumentPrintCondition& argumentPrintCondition)
{
    if (argumentPrintData.find(argumentId) != argumentPrintData.end())
    {
        argumentPrintData.erase(argumentId);
    }
    argumentPrintData.insert(std::make_pair(argumentId, std::make_pair(filePath, argumentPrintCondition)));
}

bool ArgumentPrinter::argumentPrintDataExists(const size_t argumentId) const
{
    if (argumentPrintData.find(argumentId) == argumentPrintData.end())
    {
        return false;
    }
    return true;
}

} // namespace ktt
