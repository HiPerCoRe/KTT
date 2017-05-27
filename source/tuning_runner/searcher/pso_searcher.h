#pragma once

#include <algorithm>
#include <chrono>
#include <limits>
#include <random>

#include "searcher.h"
#include "../../kernel/kernel_parameter.h"

namespace ktt
{

class PSOSearcher : public Searcher
{
public:
    PSOSearcher(const std::vector<KernelConfiguration>& configurations, const std::vector<KernelParameter>& parameters, const double fraction,
        const size_t swarmSize, const double influenceGlobal, const double influenceLocal, const double influenceRandom) :
        configurations(configurations),
        parameters(parameters),
        fraction(fraction),
        swarmSize(swarmSize),
        influenceGlobal(influenceGlobal),
        influenceLocal(influenceLocal),
        influenceRandom(influenceRandom),
        executionTimes(configurations.size(), std::numeric_limits<double>::max()),
        particleIndex(0),
        particlePositions(swarmSize),
        globalBestTime(std::numeric_limits<double>::max()),
        localBestTimes(swarmSize, std::numeric_limits<double>::max()),
        globalBestConfiguration(DimensionVector(0, 0, 0), DimensionVector(0, 0, 0), std::vector<ParameterValue>()),
        localBestConfigurations(swarmSize, KernelConfiguration(DimensionVector(0, 0, 0), DimensionVector(0, 0, 0), std::vector<ParameterValue>())),
        generator(static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count())),
        intDistribution(0, static_cast<int>(configurations.size())),
        probabilityDistribution(0.0, 1.0)
    {
        if (configurations.size() == 0)
        {
            throw std::runtime_error("Configurations vector provided for searcher is empty");
        }
        for (auto& position : particlePositions)
        {
            position = static_cast<size_t>(intDistribution(generator));
        }
        index = particlePositions.at(particleIndex);
    }

    KernelConfiguration getNextConfiguration() override
    {
        return configurations.at(index);
    }

    void calculateNextConfiguration(const double previousConfigurationDuration) override
    {
        exploredIndices.push_back(index);
        executionTimes.at(index) = previousConfigurationDuration;
        if (previousConfigurationDuration < localBestTimes.at(particleIndex))
        {
            localBestTimes.at(particleIndex) = previousConfigurationDuration;
            localBestConfigurations.at(particleIndex) = configurations.at(index);
        }
        if (previousConfigurationDuration < globalBestTime)
        {
            globalBestTime = previousConfigurationDuration;
            globalBestConfiguration = configurations.at(index);
        }
        
        auto newIndex = index;
        do
        {
            auto nextConfiguration = configurations.at(index);
            for (size_t i = 0; i < nextConfiguration.getParameterValues().size(); i++)
            {
                if (probabilityDistribution(generator) <= influenceGlobal)
                {
                    nextConfiguration.parameterValues.at(i) = globalBestConfiguration.getParameterValues().at(i);
                }
                else if (probabilityDistribution(generator) <= influenceLocal)
                {
                    nextConfiguration.parameterValues.at(i) = localBestConfigurations.at(particleIndex).getParameterValues().at(i);
                }
                else if (probabilityDistribution(generator) <= influenceRandom)
                {
                    std::uniform_int_distribution<size_t> distribution(0, parameters.at(i).getValues().size());
                    std::get<1>(nextConfiguration.parameterValues.at(i)) = parameters.at(i).getValues().at(distribution(generator));
                }
            }
            newIndex = indexFromConfiguration(nextConfiguration);
        }
        while (newIndex >= configurations.size());
        particlePositions.at(particleIndex) = newIndex;

        particleIndex++;
        if (particleIndex == swarmSize)
        {
            particleIndex = 0;
        }
        index = particlePositions[particleIndex];
    }

    size_t getConfigurationsCount() const override
    {
        return std::max(static_cast<size_t>(1), std::min(configurations.size(), static_cast<size_t>(configurations.size() * fraction)));
    }

private:
    std::vector<KernelConfiguration> configurations;
    std::vector<KernelParameter> parameters;
    double fraction;
    size_t swarmSize;
    double influenceGlobal;
    double influenceLocal;
    double influenceRandom;
    
    std::vector<double> executionTimes;
    std::vector<size_t> exploredIndices;
    size_t index;
    size_t particleIndex;
    std::vector<size_t> particlePositions;

    double globalBestTime;
    std::vector<double> localBestTimes;
    KernelConfiguration globalBestConfiguration;
    std::vector<KernelConfiguration> localBestConfigurations;

    std::default_random_engine generator;
    std::uniform_int_distribution<int> intDistribution;
    std::uniform_real_distribution<double> probabilityDistribution;

    // Helper method
    size_t indexFromConfiguration(const KernelConfiguration& target) const
    {
        size_t configurationIndex = 0;
        for (auto& configuration : configurations)
        {
            size_t matchesCount = 0;
            for (size_t i = 0; i < configuration.getParameterValues().size(); i++)
            {
                if (std::get<1>(configuration.getParameterValues().at(i)) == std::get<1>(target.getParameterValues().at(i)))
                {
                    matchesCount++;
                }
            }
            if (matchesCount == configuration.getParameterValues().size())
            {
                return configurationIndex;
            }
            configurationIndex++;
        }

        return configurationIndex;
    }
};

} // namespace ktt
