#pragma once

#include <algorithm>
#include <chrono>
#include <limits>
#include <random>
#include <stdexcept>

#include "searcher.h"

namespace ktt
{

class AnnealingSearcher : public Searcher
{
public:
    static const size_t maximumAlreadyVisitedStates = 10;
    static const size_t maximumDifferences = 3;

    AnnealingSearcher(const std::vector<KernelConfiguration>& configurations, const double fraction, const double maximumTemperature) :
        configurations(configurations),
        fraction(fraction),
        maximumTemperature(maximumTemperature),
        visitedStatesCount(0),
        currentState(0),
        neighbourState(0),
        alreadyVisistedStatesCount(0),
        generator(static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count())),
        intDistribution(0, static_cast<int>(configurations.size())),
        probabilityDistribution(0.0, 1.0)
    {
        if (configurations.size() == 0)
        {
            throw std::runtime_error("Configurations vector provided for searcher is empty");
        }
        auto initialState = static_cast<size_t>(intDistribution(generator));
        currentState = initialState;
        index = initialState;
    }

    virtual KernelConfiguration getNextConfiguration() override
    {
        visitedStatesCount++;
        return configurations.at(index);
    }

    virtual void calculateNextConfiguration(const double previousConfigurationDuration) override
    {
        if (previousConfigurationDuration > 0.0) // workaround for recursive calls
        {
            exploredIndices.push_back(currentState);
            executionTimes.at(index) = previousConfigurationDuration;
        }
        
        auto progress = visitedStatesCount / static_cast<double>(getConfigurationsCount());
        auto temperature = maximumTemperature * (1.0 - progress);

        auto acceptanceProbability = getAcceptanceProbability(executionTimes.at(currentState), executionTimes.at(neighbourState), temperature);
        auto randomProbability = probabilityDistribution(generator);
        if (acceptanceProbability > randomProbability)
        {
            currentState = neighbourState;
        }

        auto neighbours = getNeighbours(currentState);
        neighbourState = neighbours.at(static_cast<size_t>(intDistribution(generator)) % neighbours.size());

        if (executionTimes.at(neighbourState) != std::numeric_limits<double>::max())
        {
            if (alreadyVisistedStatesCount < maximumAlreadyVisitedStates)
            {
                alreadyVisistedStatesCount++;
                calculateNextConfiguration(-1.0);
                return;
            }
        }
        alreadyVisistedStatesCount = 0;
        index = neighbourState;
    }

    virtual size_t getConfigurationsCount() const override
    {
        return std::max(static_cast<size_t>(1), std::min(configurations.size(), static_cast<size_t>(configurations.size() * fraction)));
    }

private:
    std::vector<KernelConfiguration> configurations;
    size_t index;
    double fraction;
    double maximumTemperature;
    size_t visitedStatesCount;
    size_t currentState;
    size_t neighbourState;
    size_t alreadyVisistedStatesCount;

    std::vector<double> executionTimes;
    std::vector<size_t> exploredIndices;

    std::default_random_engine generator;
    std::uniform_int_distribution<int> intDistribution;
    std::uniform_real_distribution<double> probabilityDistribution;

    // Helper methods
    std::vector<size_t> getNeighbours(const size_t referenceId) const
    {
        std::vector<size_t> neighbours;
        size_t otherId = 0;
        for (const auto& configuration : configurations)
        {
            size_t differences = 0;
            size_t settingId = 0;
            for (const auto& parameter : configuration.getParameterValues())
            {
                if (std::get<1>(parameter) != std::get<1>(configurations.at(referenceId).getParameterValues().at(settingId)))
                {
                    differences++;
                }
                settingId++;
            }

            if (differences <= maximumDifferences)
            {
                neighbours.push_back(otherId);
            }
            otherId++;
        }

        if (neighbours.size() == 0)
        {
            throw std::runtime_error("Annealing searcher could not find any neighbours");
        }
        return neighbours;
    }

    double getAcceptanceProbability(const double currentEnergy, const double neighbourEnergy, const double temperature) const
    {
        if (neighbourEnergy < currentEnergy)
        {
            return 1.0;
        }
        return exp(-(neighbourEnergy - currentEnergy) / temperature);
    }
};

} // namespace ktt
