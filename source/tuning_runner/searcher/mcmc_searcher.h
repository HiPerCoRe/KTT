#pragma once

#include <algorithm>
#include <chrono>
#include <limits>
#include <random>
#include <stdexcept>
#include <set>
#include "searcher.h"

#define MCMC_VERBOSITY 0

namespace ktt
{

class MCMCSearcher : public Searcher
{
public:
    static const size_t maximumDifferences = 1;
    static const size_t bootIterations = 10;

    MCMCSearcher(const std::vector<KernelConfiguration>& configurations, const std::vector<double>& start) :
        configurations(configurations),
        visitedStatesCount(0),
        originState(0),
        currentState(0),
        executionTimes(configurations.size(), std::numeric_limits<double>::max()),
        exploredIndices(configurations.size(), false),
        generator(static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count())),
        intDistribution(0, static_cast<int>(configurations.size()-1)),
        probabilityDistribution(0.0, 1.0),
        bestTime(std::numeric_limits<double>::max())
    {
        if (configurations.size() == 0)
        {
            throw std::runtime_error("Configurations vector provided for searcher is empty");
        }

        size_t initialState;
        if (start.size() > 0) 
            initialState = searchStateIndex(start);
        else {
            initialState = static_cast<size_t>(intDistribution(generator));
            boot = bootIterations;
        }
        originState = currentState = initialState;
        index = initialState;

        for (int i = 0; i < configurations.size(); i++)
            unexploredIndices.insert(i);
    }

    void calculateNextConfiguration(const double previousDuration) override
    {
        visitedStatesCount++;
        exploredIndices[index] = true;
        unexploredIndices.erase(index);
        executionTimes.at(index) = previousDuration;

        // boot-up, sweeps randomly across bootIterations states and sets
        // origin of MCMC to the best state
        if (boot > 0) 
        {
            if (executionTimes.at(currentState) <= executionTimes.at(originState)) {            
                originState = currentState;
                bestTime = executionTimes.at(currentState);
#if MCMC_VERBOSITY > 0
                std::cout << "MCMC BOOT step " << visitedStatesCount << ": New best performance (" << bestTime << ")!\n";
#endif      
            }
            boot--;
            while (unexploredIndices.find(index) == unexploredIndices.end() 
                || unexploredIndices.empty()) 
            {
                index = static_cast<size_t>(intDistribution(generator));
            }
            currentState = index;
            return;
        }

        // acceptation of a new state
        if ((executionTimes.at(currentState) <= executionTimes.at(originState))
        || probabilityDistribution(generator) < 0.02)
        {
            originState = currentState;
#if MCMC_VERBOSITY > 0
            if (executionTimes.at(currentState) < bestTime)
                bestTime = executionTimes.at(currentState);
            if (executionTimes.at(currentState) <= executionTimes.at(originState))
                std::cout << "MCMC step " << visitedStatesCount << ": Accepting a new state (performance improvement).\n";
                if (executionTimes.at(currentState) == bestTime)
                    std::cout << "MCMC step " << visitedStatesCount << ": New best performance (" << bestTime << ")!\n";
            else
                std::cout << "MCMC step " << visitedStatesCount << ": Accepting a new state (random escape).\n";
        }
        else {
            std::cout << "MCMC step " << visitedStatesCount << ": Continuing searching neighbours.\n";
#endif
        }

        if (unexploredIndices.empty()) 
            return;

        std::vector<size_t> neighbours = getNeighbours(originState);

        // reset origin position when there are no neighbours
        if (neighbours.size() == 0)
        {
#if MCMC_VERBOSITY > 0
            std::cout << "MCMC step " << visitedStatesCount << ": No neighbours, reseting position.\n";
#endif
            while (unexploredIndices.find(originState) == unexploredIndices.end())
            {
                originState = static_cast<size_t>(intDistribution(generator));
            }
            index = currentState = originState;
            return;
        }

#if MCMC_VERBOSITY > 1
        std::cout << "MCMC step " << visitedStatesCount << ": Choosing randomly one of " << neighbours.size() << " neighbours.\n"; 
#endif
        // select a random neighbour state
        currentState = neighbours.at(static_cast<size_t>(intDistribution(generator)) % neighbours.size());
        
        index = currentState;
    }

    KernelConfiguration getCurrentConfiguration() const override
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
    std::vector<KernelConfiguration> configurations;
    size_t index;

    size_t visitedStatesCount;
    size_t originState;
    size_t currentState;
    size_t boot;

    std::vector<double> executionTimes;
    std::vector<bool> exploredIndices;
    std::set<size_t> unexploredIndices;

    std::default_random_engine generator;
    std::uniform_int_distribution<int> intDistribution;
    std::uniform_real_distribution<double> probabilityDistribution;

    double bestTime;

    // Helper methods
    std::vector<size_t> getNeighbours(const size_t referenceId) const
    {
        std::vector<size_t> neighbours;
//        for (const auto& configuration : configurations)
        for (const auto& i : unexploredIndices)
        {
            size_t differences = 0;
            size_t settingId = 0;
            for (const auto& parameter : configurations[i].getParameterPairs())
            {
                if (parameter.getValue() != configurations.at(referenceId).getParameterPairs().at(settingId).getValue())
                {
                    differences++;
                }
                settingId++;
            }

            if (differences <= maximumDifferences) 
            {
                neighbours.push_back(i);
            }
        }

        return neighbours;
    }

    size_t searchStateIndex(const std::vector<double> &state) {
        size_t states = state.size();
        size_t ret = 0;
        bool match;
        for (const auto& configuration : configurations) {
            match = true;
            for (size_t i = 0; i < states; i++) {
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
