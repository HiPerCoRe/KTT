#pragma once

#include <algorithm>
#include <chrono>
#include <limits>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuning_runner/searcher/searcher.h>
#include <utility/logger.h>

namespace ktt
{

class MCMCSearcher : public Searcher
{
public:
    static const size_t maximumDifferences = 2;
    static const size_t bootIterations = 10;
    const double escapeProbability = 0.02;

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

        for (size_t i = 0; i < configurations.size(); i++)
            unexploredIndices.insert(i);
    }

    void calculateNextConfiguration(const bool, const KernelConfiguration&, const double previousDuration, const KernelProfilingData&,
        const std::map<KernelId, KernelProfilingData>&) override
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

                std::stringstream stream;
                stream << "MCMC BOOT step " << visitedStatesCount << ": New best performance (" << bestTime << ")!";
                Logger::getLogger().log(LoggingLevel::Debug, stream.str()); 
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

        std::stringstream stream;
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
                stream << "MCMC step " << visitedStatesCount << ": Accepting a new state (performance improvement).";
                Logger::getLogger().log(LoggingLevel::Debug, stream.str());
                if (executionTimes.at(currentState) == bestTime)
                {
                    stream.clear();
                    stream << "MCMC step " << visitedStatesCount << ": New best performance (" << bestTime << ")!";
                    Logger::getLogger().log(LoggingLevel::Debug, stream.str());
                }
            }
            else
            {
                stream << "MCMC step " << visitedStatesCount << ": Accepting a new state (random escape).";
                Logger::getLogger().log(LoggingLevel::Debug, stream.str());
            }
        }
        else
        {
            stream << "MCMC step " << visitedStatesCount << ": Continuing searching neighbours.";
            Logger::getLogger().log(LoggingLevel::Debug, stream.str());
        }

        if (unexploredIndices.empty()) 
            return;

        std::vector<size_t> neighbours = getNeighbours(originState);

        // reset origin position when there are no neighbours
        if (neighbours.size() == 0)
        {
            std::stringstream debugStream;
            debugStream << "MCMC step " << visitedStatesCount << ": No neighbours, reseting position.";
            Logger::getLogger().log(LoggingLevel::Debug, debugStream.str());

            while (unexploredIndices.find(originState) == unexploredIndices.end())
            {
                originState = static_cast<size_t>(intDistribution(generator));
            }
            index = currentState = originState;
            return;
        }

        stream.clear();
        stream << "MCMC step " << visitedStatesCount << ": Choosing randomly one of " << neighbours.size() << " neighbours.";
        Logger::getLogger().log(LoggingLevel::Debug, stream.str());

        // select a random neighbour state
        currentState = neighbours.at(static_cast<size_t>(intDistribution(generator)) % neighbours.size());
        
        index = currentState;
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

    std::vector<double> dimRelevance; // relevance of each dimmension regarding to performance
    std::vector<double> dimIndependence; // independence of each dimmension

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
            for (int j = 0; j < configurations[i].getParameterPairs()) {
                if (configurations[referenceId].getParameterPairs().at(j).getValue() != configurations[i].getParameterPairs().at(j).getValue())
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
            Logger::getLogger().log(LoggingLevel::Warning, "MCMC starting point not found.");
            ret = 0;
        }

        return ret;
    }
};

} // namespace ktt
