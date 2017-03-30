#pragma once

#include "kernel_configuration.h"

namespace ktt
{

KernelConfiguration::KernelConfiguration(const DimensionVector& globalSize, const DimensionVector& localSize,
    const std::vector<ParameterValue>& parameterValues):
    globalSize(globalSize),
    localSize(localSize),
    parameterValues(parameterValues)
{}
    
DimensionVector KernelConfiguration::getGlobalSize() const
{
    return globalSize;
}

DimensionVector KernelConfiguration::getLocalSize() const
{
    return localSize;
}

std::vector<ParameterValue> KernelConfiguration::getParameterValues() const
{
    return parameterValues;
}

std::ostream& operator<<(std::ostream& outputTarget, const KernelConfiguration& kernelConfiguration)
{
    outputTarget << "global: " << std::get<0>(kernelConfiguration.globalSize) << " " << std::get<1>(kernelConfiguration.globalSize) << " "
        << std::get<2>(kernelConfiguration.globalSize) << "; ";
    outputTarget << "local: " << std::get<0>(kernelConfiguration.localSize) << " " << std::get<1>(kernelConfiguration.localSize) << " "
        << std::get<2>(kernelConfiguration.localSize) << "; ";
    outputTarget << "parameters: ";

    for (const auto& value : kernelConfiguration.parameterValues)
    {
        outputTarget << std::get<0>(value) << ": " << std::get<1>(value) << " ";
    }
    outputTarget << std::endl;

    return outputTarget;
}

} // namespace ktt
