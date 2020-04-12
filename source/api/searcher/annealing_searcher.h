/** @file annealing_searcher.h
  * Searcher which explores configurations using simulated annealing method.
  */
#pragma once

#include <algorithm>
#include <chrono>
#include <limits>
#include <random>
#include <stdexcept>
#include <api/searcher/searcher.h>

namespace ktt
{

/** @class AnnealingSearcher
  * Searcher which explores configurations using simulated annealing method.
  */
class AnnealingSearcher : public Searcher
{
public:
    static const size_t maximumAlreadyVisitedStates = 10;
    static const size_t maximumDifferences = 3;

    /** @fn AnnealingSearcher(const double maximumTemperature)
      * Initializes annealing searcher.
      * @param maximumTemperature Maximum temperature parameter for simulated annealing.
      */
    AnnealingSearcher(const double maximumTemperature) :
        configurations(nullptr),
        index(0),
        maximumTemperature(maximumTemperature),
        visitedStatesCount(0),
        currentState(0),
        neighbourState(0),
        alreadyVisistedStatesCount(0),
        generator(static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count())),
        probabilityDistribution(0.0, 1.0)
    {}

    void initializeConfigurations(const std::vector<KernelConfiguration>& configurations) override
    {
        if (configurations.empty())
        {
            throw std::runtime_error("No configurations provided for annealing searcher");
        }

        this->configurations = &configurations;
        intDistribution = std::uniform_int_distribution<size_t>(0, configurations.size() - 1),
        executionTimes.resize(configurations.size(), std::numeric_limits<double>::max());

        const size_t initialState = intDistribution(generator);
        currentState = initialState;
        index = initialState;
    }

    void calculateNextConfiguration(const ComputationResult& previousResult) override
    {
        if (previousResult.getDuration() > 0) // workaround for recursive calls
        {
            visitedStatesCount++;
            exploredIndices.push_back(currentState);
            executionTimes.at(index) = static_cast<double>(previousResult.getDuration());
        }
        
        double progress = visitedStatesCount / static_cast<double>(configurations->size());
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
                ++alreadyVisistedStatesCount;
                ComputationResult result(previousResult.getKernelName(), previousResult.getConfiguration(), 0, previousResult.getCompilationData(),
                    previousResult.getProfilingData());
                calculateNextConfiguration(result);
                return;
            }
        }

        alreadyVisistedStatesCount = 0;
        index = neighbourState;
    }

    const KernelConfiguration& getNextConfiguration() const override
    {
        return configurations->at(index);
    }

    size_t getUnexploredConfigurationCount() const override
    {
        if (visitedStatesCount >= configurations->size())
        {
            return 0;
        }

        return configurations->size() - visitedStatesCount;
    }

    bool isInitialized() const override
    {
        return configurations != nullptr;
    }

    void reset() override
    {
        configurations = nullptr;
        index = 0;
        visitedStatesCount = 0;
        currentState = 0;
        neighbourState = 0;
        alreadyVisistedStatesCount = 0;
        executionTimes.clear();
        exploredIndices.clear();
    }

private:
    const std::vector<KernelConfiguration>* configurations;
    size_t index;
    double maximumTemperature;
    size_t visitedStatesCount;
    size_t currentState;
    size_t neighbourState;
    size_t alreadyVisistedStatesCount;

    std::vector<double> executionTimes;
    std::vector<size_t> exploredIndices;

    std::default_random_engine generator;
    std::uniform_int_distribution<size_t> intDistribution;
    std::uniform_real_distribution<double> probabilityDistribution;

    // Helper methods
    std::vector<size_t> getNeighbours(const size_t referenceId) const
    {
        std::vector<size_t> neighbours;
        size_t otherId = 0;
        for (const auto& configuration : *configurations)
        {
            size_t differences = 0;
            size_t settingId = 0;
            for (const auto& parameter : configuration.getParameterPairs())
            {
                if (parameter.getValue() != configurations->at(referenceId).getParameterPairs().at(settingId).getValue())
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
