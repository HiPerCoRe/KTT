#include <string>

#include "tuning_runner.h"
#include "searcher/annealing_searcher.h"
#include "searcher/full_searcher.h"
#include "searcher/random_searcher.h"
#include "searcher/pso_searcher.h"

namespace ktt
{

TuningRunner::TuningRunner(ArgumentManager* argumentManager, KernelManager* kernelManager, OpenCLCore* openCLCore) :
    argumentManager(argumentManager),
    kernelManager(kernelManager),
    openCLCore(openCLCore)
{}

std::vector<TuningResult> TuningRunner::tuneKernel(const size_t id)
{
    if (id >= kernelManager->getKernelCount())
    {
        throw std::runtime_error(std::string("Invalid kernel id: " + std::to_string(id)));
    }
    resultValidator.clearReferenceResults(id);

    std::vector<TuningResult> results;
    const Kernel* kernel = kernelManager->getKernel(id);
    std::unique_ptr<Searcher> searcher = getSearcher(kernel->getSearchMethod(), kernel->getSearchArguments(),
        kernelManager->getKernelConfigurations(id), kernel->getParameters());
    size_t configurationsCount = searcher->getConfigurationsCount();

    for (size_t i = 0; i < configurationsCount; i++)
    {
        KernelConfiguration currentConfiguration = searcher->getNextConfiguration();
        std::string source = kernelManager->getKernelSourceWithDefines(id, currentConfiguration);

        KernelRunResult result;
        try
        {
            std::cout << "Launching kernel <" << kernel->getName() << "> with configuration: " << currentConfiguration << std::endl;
            result = openCLCore->runKernel(source, kernel->getName(), convertDimensionVector(currentConfiguration.getGlobalSize()),
                convertDimensionVector(currentConfiguration.getLocalSize()), getKernelArguments(id));
        }
        catch (const std::runtime_error& error)
        {
            std::cerr << "Kernel run execution failed for configuration: " << std::endl << currentConfiguration << std::endl;
            std::cerr << error.what() << std::endl;
        }

        searcher->calculateNextConfiguration(static_cast<double>(result.getDuration()));
        if (result.getDuration() != 0)
        {
            bool storeResult = true;
            if (kernel->hasReferenceClass() || kernel->hasReferenceKernel())
            {
                storeResult = validateResult(kernel, result);
            }
            if (storeResult)
            {
                results.emplace_back(TuningResult(kernel->getName(), result.getDuration(), currentConfiguration));
            }
        }
    }

    return results;
}

void TuningRunner::setValidationMethod(const ValidationMethod& validationMethod, const double toleranceThreshold)
{
    resultValidator.setValidationMethod(validationMethod);
    resultValidator.setToleranceThreshold(toleranceThreshold);
}

std::unique_ptr<Searcher> TuningRunner::getSearcher(const SearchMethod& searchMethod, const std::vector<double>& searchArguments,
    const std::vector<KernelConfiguration>& configurations, const std::vector<KernelParameter>& parameters) const
{
    std::unique_ptr<Searcher> searcher;

    switch (searchMethod)
    {
    case SearchMethod::FullSearch:
        searcher.reset(new FullSearcher(configurations));
        break;
    case SearchMethod::RandomSearch:
        searcher.reset(new RandomSearcher(configurations, searchArguments.at(0)));
        break;
    case SearchMethod::PSO:
        searcher.reset(new PSOSearcher(configurations, parameters, searchArguments.at(0), static_cast<size_t>(searchArguments.at(1)),
            searchArguments.at(2), searchArguments.at(3), searchArguments.at(4)));
        break;
    default:
        searcher.reset(new AnnealingSearcher(configurations, searchArguments.at(0), searchArguments.at(1)));
    }

    return searcher;
}

std::vector<size_t> TuningRunner::convertDimensionVector(const DimensionVector& vector) const
{
    std::vector<size_t> result;

    result.push_back(std::get<0>(vector));
    result.push_back(std::get<1>(vector));
    result.push_back(std::get<2>(vector));

    return result;
}

std::vector<KernelArgument> TuningRunner::getKernelArguments(const size_t kernelId) const
{
    std::vector<KernelArgument> result;

    std::vector<size_t> argumentIndices = kernelManager->getKernel(kernelId)->getArgumentIndices();
    
    for (const auto index : argumentIndices)
    {
        result.emplace_back(argumentManager->getArgument(index));
    }

    return result;
}

bool TuningRunner::validateResult(const Kernel* kernel, const KernelRunResult& result)
{
    bool validationResult = true;

    if (kernel->hasReferenceClass())
    {
        validationResult &= validateResultWithClass(kernel, result);
    }

    if (kernel->hasReferenceKernel())
    {
        validationResult &= validateResultWithKernel(kernel, result);
    }

    return validationResult;
}

bool TuningRunner::validateResultWithClass(const Kernel* kernel, const KernelRunResult& result)
{
    size_t classArgumentId = kernel->getResultArgumentIdForClass();

    if (!argumentIndexExists(classArgumentId, kernel->getArgumentIndices()))
    {
        throw std::runtime_error(std::string("Reference class argument with following id is not associated with given kernel: " +
            std::to_string(classArgumentId)));
    }
    if (argumentManager->getArgument(classArgumentId).getArgumentMemoryType() == ArgumentMemoryType::ReadOnly)
    {
        throw std::runtime_error(std::string("Reference class argument with following id is marked as read only: " +
            std::to_string(classArgumentId)));
    }

    if (!resultValidator.hasReferenceClassResult(kernel->getId()))
    {
        resultValidator.setReferenceClassResult(kernel->getId(), getReferenceResultFromClass(kernel->getReferenceClass(), classArgumentId));
    }

    size_t argumentIndexInResult;
    const auto& resultArguments = result.getResultArguments();

    for (size_t i = 0; i < resultArguments.size(); i++)
    {
        if (resultArguments.at(i).getId() == classArgumentId)
        {
            argumentIndexInResult = i;
            break;
        }
    }

    bool validationResult = resultValidator.validateArgumentWithClass(kernel->getId(), resultArguments.at(argumentIndexInResult));
    return validationResult;
}

bool TuningRunner::validateResultWithKernel(const Kernel* kernel, const KernelRunResult& result)
{
    auto indices = kernel->getArgumentIndices();
    auto referenceIndices = kernel->getResultArgumentIds();
    size_t referenceKernelId = kernel->getReferenceKernelId();

    for (const auto argumentId : referenceIndices)
    {
        if (!argumentIndexExists(argumentId, indices))
        {
            throw std::runtime_error(std::string("Reference kernel argument with following id is not associated with given kernel: " +
                std::to_string(argumentId)));
        }

        if (argumentManager->getArgument(argumentId).getArgumentMemoryType() == ArgumentMemoryType::ReadOnly)
        {
            throw std::runtime_error(std::string("Reference kernel argument with following id is marked as read only: " +
                std::to_string(argumentId)));
        }
    }

    if (!resultValidator.hasReferenceKernelResult(kernel->getId()))
    {
        resultValidator.setReferenceKernelResult(kernel->getId(), getReferenceResultFromKernel(referenceKernelId,
            kernel->getReferenceKernelConfiguration()));
    }

    std::vector<size_t> argumentIndicesInResult;
    const auto& resultArguments = result.getResultArguments();

    for (size_t i = 0; i < resultArguments.size(); i++)
    {
        if (argumentIndexExists(resultArguments.at(i).getId(), referenceIndices))
        {
            argumentIndicesInResult.push_back(i);
        }
    }

    std::vector<KernelArgument> argumentsToValidate;
    for (const auto index : argumentIndicesInResult)
    {
        argumentsToValidate.push_back(resultArguments.at(index));
    }

    bool validationResult = resultValidator.validateArgumentWithKernel(kernel->getId(), argumentsToValidate);
    return validationResult;
}

bool TuningRunner::argumentIndexExists(const size_t argumentIndex, const std::vector<size_t>& argumentIndices) const
{
    for (const auto index : argumentIndices)
    {
        if (index == argumentIndex)
        {
            return true;
        }
    }
    return false;
}

KernelArgument TuningRunner::getReferenceResultFromClass(const ReferenceClass* referenceClass, const size_t referenceArgumentId) const
{
    size_t dataSize = referenceClass->getDataSizeInBytes();
    ArgumentDataType dataType = referenceClass->getDataType();

    if (dataType == ArgumentDataType::Double)
    {
        double* buffer = static_cast<double*>(referenceClass->getData());
        std::vector<double> vectorDouble(buffer, buffer + dataSize / sizeof(double));
        
        if (vectorDouble.size() == 0)
        {
            throw std::runtime_error(std::string("Data provided by reference class for argument with following id is empty: " +
                std::to_string(referenceArgumentId)));
        }

        ArgumentQuantity quantity = vectorDouble.size() == 1 ? ArgumentQuantity::Scalar : ArgumentQuantity::Vector;
        return KernelArgument(referenceArgumentId, vectorDouble, ArgumentMemoryType::ReadWrite, quantity);
    }
    else if (dataType == ArgumentDataType::Float)
    {
        float* buffer = static_cast<float*>(referenceClass->getData());
        std::vector<float> vectorFloat(buffer, buffer + dataSize / sizeof(float));

        if (vectorFloat.size() == 0)
        {
            throw std::runtime_error(std::string("Data provided by reference class for argument with following id is empty: " +
                std::to_string(referenceArgumentId)));
        }

        ArgumentQuantity quantity = vectorFloat.size() == 1 ? ArgumentQuantity::Scalar : ArgumentQuantity::Vector;
        return KernelArgument(referenceArgumentId, vectorFloat, ArgumentMemoryType::ReadWrite, quantity);
    }
    else
    {
        int* buffer = static_cast<int*>(referenceClass->getData());
        std::vector<int> vectorInt(buffer, buffer + dataSize / sizeof(int));

        if (vectorInt.size() == 0)
        {
            throw std::runtime_error(std::string("Data provided by reference class for argument with following id is empty: " +
                std::to_string(referenceArgumentId)));
        }

        ArgumentQuantity quantity = vectorInt.size() == 1 ? ArgumentQuantity::Scalar : ArgumentQuantity::Vector;
        return KernelArgument(referenceArgumentId, vectorInt, ArgumentMemoryType::ReadWrite, quantity);
    }
}

std::vector<KernelArgument> TuningRunner::getReferenceResultFromKernel(const size_t referenceKernelId,
    const std::vector<ParameterValue>& referenceKernelConfiguration) const
{
    const Kernel* referenceKernel = kernelManager->getKernel(referenceKernelId);
    KernelConfiguration configuration = kernelManager->getKernelConfiguration(referenceKernelId, referenceKernelConfiguration);
    std::string source = kernelManager->getKernelSourceWithDefines(referenceKernelId, configuration);

    auto result = openCLCore->runKernel(source, referenceKernel->getName(), convertDimensionVector(configuration.getGlobalSize()),
        convertDimensionVector(configuration.getLocalSize()), getKernelArguments(referenceKernelId));
    return result.getResultArguments();
}

} // namespace ktt
