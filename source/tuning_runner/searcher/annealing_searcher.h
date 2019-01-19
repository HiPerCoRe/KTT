#pragma once

#include <algorithm>
#include <chrono>
#include <limits>
#include <random>
#include <stdexcept>
#include <tuning_runner/searcher/searcher.h>

namespace ktt
{

class AnnealingSearcher : public Searcher
{
public:
    static const size_t maximumAlreadyVisitedStates = 10;
    static const size_t maximumDifferences = 3;

    AnnealingSearcher(const std::vector<KernelConfiguration>& configurations, const double maximumTemperature) :
        configurations(configurations),
        maximumTemperature(maximumTemperature),
        visitedStatesCount(0),
        currentState(0),
        neighbourState(0),
        alreadyVisistedStatesCount(0),
        executionTimes(configurations.size(), std::numeric_limits<double>::max()),
        generator(static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count())),
        intDistribution(0, static_cast<int>(configurations.size()-1)),
        probabilityDistribution(0.0, 1.0)
    {
        if (configurations.size() == 0)
        {
            throw std::runtime_error("Configurations vector provided for searcher is empty");
        }
        size_t initialState = static_cast<size_t>(intDistribution(generator));
        currentState = initialState;
        index = initialState;
    }

    void calculateNextConfiguration(const bool successFlag, const KernelConfiguration& previousConfiguration, const double previousDuration,
        const KernelProfilingData& previousProfilingData, const std::map<KernelId, KernelProfilingData>& previousCompositionProfilingData) override
    {
        if (previousDuration >= 0.0) // workaround for recursive calls
        {
            visitedStatesCount++;
            exploredIndices.push_back(currentState);
            executionTimes.at(index) = previousDuration;
        }
        
        double progress = visitedStatesCount / static_cast<double>(configurations.size());
        double temperature = maximumTemperature * (1.0 - progress);

        double acceptanceProbability = getAcceptanceProbability(executionTimes.at(currentState), executionTimes.at(neighbourState), temperature);
        double randomProbability = probabilityDistribution(generator);
        if (acceptanceProbability > randomProbability)
        {
            currentState = neighbourState;
        }

        std::vector<size_t> neighbours = getNeighbours(currentState);
        neighbourState = neighbours.at(static_cast<size_t>(intDistribution(generator)) % neighbours.size());

        if (executionTimes.at(neighbourState) != std::numeric_limits<double>::max())
        {
            if (alreadyVisistedStatesCount < maximumAlreadyVisitedStates)
            {
                alreadyVisistedStatesCount++;
                calculateNextConfiguration(successFlag, previousConfiguration, -1.0, previousProfilingData, previousCompositionProfilingData);
                return;
            }
        }
        alreadyVisistedStatesCount = 0;
        index = neighbourState;
    }

    KernelConfiguration getNextConfiguration() const override
    {
        return configurations.at(index);
    }

    size_t getUnexploredConfigurationCount() const override
    {
        if (visitedStatesCount >= configurations.size())
        {
            return 0;
        }

        return configurations.size() - visitedStatesCount;
    }

private:
    const std::vector<KernelConfiguration>& configurations;
    size_t index;
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
            for (const auto& parameter : configuration.getParameterPairs())
            {
                if (parameter.getValue() != configurations.at(referenceId).getParameterPairs().at(settingId).getValue())
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
        return std::exp(-(neighbourEnergy - currentEnergy) / temperature);
    }
};

} // namespace ktt
