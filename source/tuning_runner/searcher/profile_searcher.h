#pragma once

#include <algorithm>
#include <random>
#include <iostream>
#include <fstream>
#include <tuning_runner/searcher/searcher.h>

namespace ktt
{

class ProfileSearcher : public Searcher
{
public:
    ProfileSearcher(const std::vector<KernelConfiguration>& configurations, const double computeCapability) :
        configurations(configurations)
    {
        if (configurations.size() == 0)
        {
            throw std::runtime_error("Configurations vector provided for searcher is empty");
        }

        std::ofstream profilingFile;
        profilingFile.open ("ktt-tempfile-conf.csv");
        const int pars = configurations[0].getParameterPairs().size();
        for (int i = 0; i < pars; i++) {
            profilingFile << configurations[0].getParameterPairs()[i].getName();
            if (i < pars-1) profilingFile << ",";
        }
        profilingFile << std::endl;
        for (auto conf : configurations) {
            for (int i = 0; i < pars; i++) {
                profilingFile << conf.getParameterPairs()[i].getValue();
                if (i < pars-1) profilingFile << ",";
            }
            profilingFile << std::endl;
        }
        profilingFile.close();

        this->computeCapability = computeCapability;
    }

    void calculateNextConfiguration(const KernelResult& kernelResult) override
    {
        std::vector<KernelProfilingCounter> counters = kernelResult.getProfilingData().getAllCounters(); //getCounter("name")
        //KernelResult.getCompilationData();

        std::ofstream profilingFile;
        profilingFile.open("ktt-tempfile-pc.csv");

        const int cnt = counters.size();
        for (int i = 0; i < cnt; i++) {
            profilingFile << counters[i].getName();
            if (i < cnt-1) profilingFile << ",";
        }
        profilingFile << std::endl;
        for (int i = 0; i < cnt; i++) {
            switch(counters[i].getType()) {
            case ktt::ProfilingCounterType::Int:
                profilingFile << counters[i].getValue().intValue;
                break;
            case ktt::ProfilingCounterType::UnsignedInt:
                profilingFile << counters[i].getValue().uintValue;
                break;
            case ktt::ProfilingCounterType::Percent:
                profilingFile << counters[i].getValue().percentValue;
                break;
            case ktt::ProfilingCounterType::Throughput:
                profilingFile << counters[i].getValue().throughputValue;
                break;
            case ktt::ProfilingCounterType::UtilizationLevel:
                profilingFile << counters[i].getValue().utilizationLevelValue;
                break;
            case ktt::ProfilingCounterType::Double:
                profilingFile << counters[i].getValue().doubleValue;
                break;
            default:
                throw std::runtime_error("Unknown type of profiling counter.");
                break;
            }
            if (i < cnt-1) profilingFile << ",";
        }
        profilingFile.close();
        std::string command = "./ktt-profiling-searcher.py -o ktt-tempfile-conf.csv --oc " + std::to_string(computeCapability) + " -s ../../../profilbased-searcher/data-reducedcounters/750-gemm-reduced --sc 5.0 -i " + std::to_string(index) + " -p ktt-tempfile-pc.csv";
        //std::cout << command << std::endl;
        system(command.c_str());
        std::fstream indexFile("ktt-tempfile-idx.dat", std::fstream::in);
        indexFile >> index;
        indexFile.close();
        //std::cout << "Index: " << index << std::endl;
    }

    KernelConfiguration getNextConfiguration() const override
    {
        return configurations.at(index);
    }

    size_t getUnexploredConfigurationCount() const override
    {
        if (index >= configurations.size())
        {
            return 0;
        }

        return configurations.size() - index;
    }

private:
    const std::vector<KernelConfiguration>& configurations;
    size_t index;
    double computeCapability;
};

} // namespace ktt
