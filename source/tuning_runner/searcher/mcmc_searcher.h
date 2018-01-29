#pragma once

#include <algorithm>
#include <chrono>
#include <limits>
#include <random>
#include <stdexcept>
#include "searcher.h"

namespace ktt
{

class MCMCSearcher : public Searcher
{
public:
    static const size_t maximumDifferences = 3;

    MCMCSearcher(const std::vector<KernelConfiguration>& configurations,  const double fraction, const std::vector<double> start) :
        configurations(configurations),
        fraction(fraction),
        visitedStatesCount(0),
        originState(0),
        currentState(0),
        executionTimes(configurations.size(), std::numeric_limits<double>::max()),
        exploredIndices(configurations.size(), false),
        generator(static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count())),
        intDistribution(0, static_cast<int>(configurations.size())),
        probabilityDistribution(0.0, 1.0)
    {
        if (configurations.size() == 0)
        {
            throw std::runtime_error("Configurations vector provided for searcher is empty");
        }

        size_t initialState;
        if (start.size()) 
            initialState = searchStateIndex(start);
        else
            initialState = static_cast<size_t>(intDistribution(generator));
        originState = currentState = initialState;
        index = initialState;
    }

    void calculateNextConfiguration(const double previousDuration) override
    {
        visitedStatesCount++;
        exploredIndices[index] = true;
        executionTimes.at(index) = previousDuration;

        // acceptation of a new state
        std::cout << "Exec times " << executionTimes.at(originState) << " -> " << executionTimes.at(currentState) << "\n";
        if ((executionTimes.at(currentState) <= executionTimes.at(originState))
        || probabilityDistribution(generator) < 0.05)
        {
            originState = currentState;
            std::cout << "Accepting a new state.\n";
        }

        std::vector<size_t> neighbours = getNeighbours(originState);

        // reset origin position when there are no neighbours
        if (neighbours.size() == 0)
        {
            std::cout << "No neighbours, reseting position.\n";
            while (neighbours.size() == 0)
            {
                originState = static_cast<size_t>(intDistribution(generator));
                neighbours = getNeighbours(currentState);
            }
            index = currentState = originState;
            exploredIndices[index] = true;
            return;
        }

        std::cout << "Choosing randly one of " << neighbours.size() << " neighbours.\n"; 
        // select a random neighbour state
        currentState = neighbours.at(static_cast<size_t>(intDistribution(generator)) % neighbours.size());
        
        index = currentState;
    }

    KernelConfiguration getCurrentConfiguration() const override
    {
        return configurations.at(index);
    }

    size_t getConfigurationCount() const override
    {
        return std::max(static_cast<size_t>(1), std::min(configurations.size(), static_cast<size_t>(configurations.size() * fraction)));
    }

    size_t getUnexploredConfigurationCount() const override
    {
        return getConfigurationCount() - visitedStatesCount;
    }

private:
    std::vector<KernelConfiguration> configurations;
    size_t index;
    double fraction;
    size_t visitedStatesCount;
    size_t originState;
    size_t currentState;

    std::vector<double> executionTimes;
    std::vector<bool> exploredIndices;

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

            if ((differences <= maximumDifferences) 
            && !(exploredIndices[otherId]))
            {
                neighbours.push_back(otherId);
            }
            otherId++;
        }

        return neighbours;
    }

    size_t searchStateIndex(const std::vector<double> &state) {
        int states = state.size();
        size_t ret = 0;
        bool match;
        for (const auto& configuration : configurations) {
            match = true;
            for (int i = 0; i < states; i++) {
                if (configuration.getParameterPairs().at(i).getValue() != state[i]) {
                    match = false;
                    break;
                }
            }
            if (match)
                break;
            ret++;
        }

        if (!match) {
            std::cerr << "WARNING, MCMC starting point not found." << std::endl;
            ret = 0;
        }

        return ret;
    }
};

} // namespace ktt
