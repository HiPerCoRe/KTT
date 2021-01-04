/** @file mcmc_searcher.h
  * Searcher which explores configurations using Markov chain Monte Carlo method.
  */
#pragma once

#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <api/searcher/searcher.h>

namespace ktt
{

/** @class MCMCSearcher
  * Searcher which explores configurations using Markov chain Monte Carlo method.
  */
class MCMCSearcher : public Searcher
{
public:
    /** @fn MCMCSearcher(const std::vector<double>& start)
      * Initializes MCMC searcher.
      * @param start Optional parameter which specifies starting point for MCMC searcher.
      */
    MCMCSearcher(const std::vector<double>& start) :
        Searcher(),
        index(0),
        visitedStatesCount(0),
        originState(0),
        currentState(0),
        boot(0),
        start(start),
        generator(static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count())),
        probabilityDistribution(0.0, 1.0),
        bestTime(std::numeric_limits<double>::max())
    {}

    void onInitialize() override
    {
        intDistribution = std::uniform_int_distribution<size_t>(0, getConfigurations().size() - 1),
        executionTimes.resize(getConfigurations().size(), std::numeric_limits<double>::max());
        exploredIndices.resize(getConfigurations().size(), false);

        size_t initialState = 0;

        if (!start.empty())
        {
            initialState = searchStateIndex(start);
        } 
        else
        {
            initialState = intDistribution(generator);
            boot = bootIterations;
        }

        originState = initialState;
        currentState = initialState;
        index = initialState;

        for (size_t i = 0; i < getConfigurations().size(); ++i)
        {
            unexploredIndices.insert(i);
        } 
    }

    void onReset() override
    {
        index = 0;
        visitedStatesCount = 0;
        originState = 0;
        currentState = 0;
        bestTime = std::numeric_limits<double>::max();
        executionTimes.clear();
        exploredIndices.clear();
        unexploredIndices.clear();
    }

    void calculateNextConfiguration(const ComputationResult& previousResult) override
    {
        visitedStatesCount++;
        exploredIndices[index] = true;
        unexploredIndices.erase(index);
        executionTimes.at(index) = static_cast<double>(previousResult.getDuration());

        // boot-up, sweeps randomly across bootIterations states and sets
        // origin of MCMC to the best state
        if (boot > 0) 
        {
            if (executionTimes.at(currentState) <= executionTimes.at(originState)) {            
                originState = currentState;
                bestTime = executionTimes.at(currentState);

                std::cout << "MCMC BOOT step " << visitedStatesCount << ": New best performance (" << bestTime << ")!";
            }
            boot--;
            while (unexploredIndices.find(index) == unexploredIndices.end() 
                || unexploredIndices.empty()) 
            {
                index = intDistribution(generator);
            }
            currentState = index;
            return;
        }

        // acceptation of a new state
        if ((executionTimes.at(currentState) <= executionTimes.at(originState))
        || probabilityDistribution(generator) < escapeProbability)
        {
            originState = currentState;
            
            if (executionTimes.at(currentState) < bestTime)
            {
                bestTime = executionTimes.at(currentState);
            }
            if (executionTimes.at(currentState) <= executionTimes.at(originState))
            {
                std::cout << "MCMC step " << visitedStatesCount << ": Accepting a new state (performance improvement).";

                if (executionTimes.at(currentState) == bestTime)
                {
                    std::cout << "MCMC step " << visitedStatesCount << ": New best performance (" << bestTime << ")!";
                }
            }
            else
            {
                std::cout << "MCMC step " << visitedStatesCount << ": Accepting a new state (random escape).";
            }
        }
        else
        {
            std::cout << "MCMC step " << visitedStatesCount << ": Continuing searching neighbours.";
        }

        if (unexploredIndices.empty()) 
            return;

        std::vector<size_t> neighbours = getNeighbours(originState);

        // reset origin position when there are no neighbours
        if (neighbours.size() == 0)
        {
            std::cout << "MCMC step " << visitedStatesCount << ": No neighbours, reseting position.";

            while (unexploredIndices.find(originState) == unexploredIndices.end())
            {
                originState = intDistribution(generator);
            }
            index = currentState = originState;
            return;
        }

        std::cout << "MCMC step " << visitedStatesCount << ": Choosing randomly one of " << neighbours.size() << " neighbours.";

        // select a random neighbour state
        currentState = neighbours.at(intDistribution(generator) % neighbours.size());
        
        index = currentState;
    }

    const KernelConfiguration& getNextConfiguration() const override
    {
        return getConfigurations().at(index);
    }

    size_t getUnexploredConfigurationCount() const override
    {
        if (visitedStatesCount >= getConfigurations().size())
        {
            return 0;
        }

        return getConfigurations().size() - visitedStatesCount;
    }

private:
    static const size_t maximumDifferences = 2;
    static const size_t bootIterations = 10;
    const double escapeProbability = 0.02;

    size_t index;

    size_t visitedStatesCount;
    size_t originState;
    size_t currentState;
    size_t boot;

    std::vector<double> start;
    std::vector<double> executionTimes;
    std::vector<bool> exploredIndices;
    std::set<size_t> unexploredIndices;

    std::default_random_engine generator;
    std::uniform_int_distribution<size_t> intDistribution;
    std::uniform_real_distribution<double> probabilityDistribution;

    std::vector<double> dimRelevance; // relevance of each dimension regarding to performance
    std::vector<double> dimIndependence; // independence of each dimension

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
            for (const auto& parameter : getConfigurations().at(i).getParameterPairs())
            {
                if (parameter.getValue() != getConfigurations().at(referenceId).getParameterPairs().at(settingId).getValue())
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

    /*size_t getNeighbour(const size_t referenceId) {
        // get all neighbours
        std::vector<size_t> neighbours = getNeighbours(referenceId);

        if (neighbours.size() == 0)
            return MAX_SIZET;

        std::vector<double> probabilityDistrib(neighbours.size());
        double probabSum = 0.0;

        // iterate over neighbours and assign them probability
        for (size_t i = 0; i < neighbours.size(); i++) {
            double actProbab = 0.0;
            for (int j = 0; j < getConfigurations()[i].getParameterPairs()) {
                if (getConfigurations()[referenceId].getParameterPairs().at(j).getValue() != getConfigurations()[i].getParameterPairs().at(j).getValue())
                    actProbab += dimRelevance;
            }
            probabilityDistrib[i] = actProbab;
            probabSum += actProbab;
        }
        
        // select configuration according to probability
        double random = probabilityDistribution(generator) * probabSum;
        double lastDistrib = 0.0;
        for (size_t i = 0; i < neighbours.size(); i++) {
            if (random > lastDistrib || random <= probabilityDistrib[i])
                return neighbours[i];
            lastDistrib = probabilityDistrib[i];
        }
        std::cerr << "Something horrible but recoverable happend in MCMC!" << std::endl;
        return neighbours[0];
    }*/

    size_t searchStateIndex(const std::vector<double>& state)
    {
        size_t states = state.size();
        size_t ret = 0;
        bool match = true;
        for (const auto& configuration : getConfigurations()) {
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
            std::cout << "MCMC starting point not found.";
            ret = 0;
        }

        return ret;
    }
};

} // namespace ktt
