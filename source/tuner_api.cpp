#include <iostream>

#include "tuner_api.h"

namespace ktt
{

Tuner::Tuner():
    tunerCore(new TunerCore())
{}

size_t Tuner::addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize,
    const DimensionVector& localSize)
{
    return tunerCore->getKernelManager()->addKernel(source, kernelName, globalSize, localSize);
}

size_t Tuner::addKernelFromFile(const std::string& filename, const std::string& kernelName, const DimensionVector& globalSize,
    const DimensionVector& localSize)
{
    return tunerCore->getKernelManager()->addKernelFromFile(filename, kernelName, globalSize, localSize);
}

void Tuner::addParameter(const size_t id, const KernelParameter& parameter)
{
    try
    {
        tunerCore->getKernelManager()->getKernel(id)->addParameter(parameter);
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
    }
}

void Tuner::addArgumentInt(const size_t id, const std::vector<int>& data)
{
    try
    {
        tunerCore->getKernelManager()->getKernel(id)->addArgumentInt(data);
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
    }
}

void Tuner::addArgumentFloat(const size_t id, const std::vector<float>& data)
{
    try
    {
        tunerCore->getKernelManager()->getKernel(id)->addArgumentFloat(data);
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
    }
}

void Tuner::addArgumentDouble(const size_t id, const std::vector<double>& data)
{
    try
    {
        tunerCore->getKernelManager()->getKernel(id)->addArgumentDouble(data);
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
    }
}

void Tuner::useSearchMethod(const size_t id, const SearchMethod& searchMethod, const std::vector<double>& searchArguments)
{
    try
    {
        tunerCore->getKernelManager()->getKernel(id)->useSearchMethod(searchMethod, searchArguments);
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
    }
}

} // namespace ktt
