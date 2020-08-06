#pragma once

#include <algorithm>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <api/searcher/searcher.h>

#define PROFILESEARCHER_TEMPFILE_CONF   "ktt-tempfile-conf.csv"
#define PROFILESEARCHER_TEMPFILE_PC     "ktt-tempfile-pc.csv"
#define PROFILESEARCHER_TEMPFILE_IDX    "ktt-tempfile-idx.dat"

namespace ktt
{

class ProfileSearcher : public Searcher
{
public:
    ProfileSearcher(const double myCC, const std::string stat, const double statCC, const std::string scratch = "") :
        Searcher(),
        myComputeCapability(myCC),
        statPrefix(stat),
        statComputeCapability(statCC),
        scratchPrefix(scratch),
        profileRequired(true)
    {}

    void onInitialize() override
    {
        // Create CSV with all configurations in the tuning space
        std::ofstream profilingFile;
        profilingFile.open (scratchPrefix + PROFILESEARCHER_TEMPFILE_CONF);
        const int pars = getConfigurations()[0].getParameterPairs().size();
        for (int i = 0; i < pars; i++) {
            profilingFile << getConfigurations()[0].getParameterPairs()[i].getName();
            if (i < pars-1) profilingFile << ",";
        }
        profilingFile << std::endl;
        for (auto conf : getConfigurations()) {
            for (int i = 0; i < pars; i++) {
                profilingFile << conf.getParameterPairs()[i].getValue();
                if (i < pars-1) profilingFile << ",";
            }
            profilingFile << std::endl;
        }
        profilingFile.close();

        // store local and global sizes
        for (auto conf : getConfigurations()) {
            globalSizes.push_back(conf.getGlobalSize());
	    localSizes.push_back(conf.getLocalSize());
        }

        // set random beginning
        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_int_distribution<> distribution(0, getConfigurations().size()-1);
        bestIdxInBatch = static_cast<size_t>(distribution(generator));
        profileRequired = true;

        // create pipes to communicate with python script

        if (::pipe(pipe_cpp_to_py) || ::pipe(pipe_py_to_cpp))
        {
          std::cout << "Couldn't open pipes" << std::endl;
          ::exit(1);
        }

        pid_t pid = fork();

        if ( pid == 0 )
        {
          //child process, i.e. python script
          // close unnecessary file descriptors
          ::close(pipe_py_to_cpp[0]);
          ::close(pipe_cpp_to_py[1]);
          std::ostringstream oss;

          // start the python script and let it load the tuning space
          std::string command = "python " + scratchPrefix + " ktt-profiling-searcher.py -o " + PROFILESEARCHER_TEMPFILE_CONF + " --oc " + std::to_string(myComputeCapability) + " --mp 46 --co 2944" + " --kb " + statPrefix + "_output_Proposed.sav --ic " + std::to_string(statComputeCapability) + " -i " + std::to_string(bestIdxInBatch) + " -p " + PROFILESEARCHER_TEMPFILE_PC + " --compute_bound";
          std::cout << command << std::endl;

          oss << "export PY_READ_FD=" << pipe_cpp_to_py[0] << " && "
            << "export PY_WRITE_FD=" << pipe_py_to_cpp[1] << " && "
            << "export PYTHONUNBUFFERED=true && " // Force stdin, stdout and stderr to be totally unbuffered.
            << command;


          ::system(oss.str().c_str());

          //after the script finishes (which happens when it receives a message "quit")
          //close the descriptors and exit the child process
          ::close(pipe_py_to_cpp[1]);
          ::close(pipe_cpp_to_py[0]);
          ::exit(0);

        }
        else if ( pid < 0 )
        {
          //error
          std::cout << "Fork failed." << std::endl;
          ::exit(1);
        }
        else
        {
          //parent process
          // close unnecessary file descriptors
          ::close(pipe_py_to_cpp[1]);
          ::close(pipe_cpp_to_py[0]);
        }
    }

    void onReset() override
    {
      //we are done, send message "quit" to python script
      std::string messageToBeSent = "quit";
      std::cout << "Writing message for python " <<  messageToBeSent << std::endl;
      ::write(pipe_cpp_to_py[1], messageToBeSent.c_str(), messageToBeSent.size());
      //close the pipes
      ::close(pipe_py_to_cpp[0]);
      ::close(pipe_cpp_to_py[1]);
    }

    ~ProfileSearcher()
    {
      //we are done, send message "quit" to python script
      std::string messageToBeSent = "quit";
      std::cout << "Writing message for python " <<  messageToBeSent << std::endl;
      ::write(pipe_cpp_to_py[1], messageToBeSent.c_str(), messageToBeSent.size());
      //close the pipes
      ::close(pipe_py_to_cpp[0]);
      ::close(pipe_cpp_to_py[1]);
    }

    void calculateNextConfiguration(const ComputationResult& computationResult) override
    {
        std::cout << "calculateNextConfiguration\n";
        if ((indices.size() > 0) && (computationResult.getDuration() < bestBatchDuration)) {
            bestBatchDuration = computationResult.getDuration();
            bestIdxInBatch = indices.back();
            std::cout << "Index " << bestIdxInBatch << " has best duration " << bestBatchDuration << "\n";
        }

        if (indices.size() == 0) {
            //std::vector<DimensionVector> globalSizes = computationResult.getConfiguration().getGlobalSizes();
            std::vector<KernelProfilingCounter> counters = computationResult.getProfilingData().getAllCounters(); //getCounter("name")
            //KernelResult.getCompilationData();

            // create CSV with actual profiling counters
            std::ofstream profilingFile;
            profilingFile.open(scratchPrefix + PROFILESEARCHER_TEMPFILE_PC);

            profilingFile << "Global size,Local size,";
	        const int cnt = counters.size();
            for (int i = 0; i < cnt; i++) {
                profilingFile << counters[i].getName();
                if (i < cnt-1) profilingFile << ",";
            }
            profilingFile << std::endl;
    	    profilingFile << globalSizes[bestIdxInBatch].getTotalSize() << ",";
	        profilingFile << localSizes[bestIdxInBatch].getTotalSize() << ",";
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

            //file is ready, send message "read <bestIdxInBatch>" to python script
            std::string messageToBeSent = "read " + std::to_string(bestIdxInBatch);
            std::cout << "-------------------- Writing message for python " <<  messageToBeSent << std::endl;
            ::write(pipe_cpp_to_py[1], messageToBeSent.c_str(), messageToBeSent.size());

            // read result of the script
            std::fstream indexFile(PROFILESEARCHER_TEMPFILE_IDX, std::fstream::in);
            while (!indexFile.eof()) {
                size_t idx;
                indexFile >> idx;
                //std::cout << "loaded idx = " << idx << std::endl;
                indices.push_back(idx); 
            }
            indexFile.close();
            indices.pop_back(); // the last element is readed twice from some weird reason

            bestIdxInBatch = indices[0];
            bestBatchDuration = std::numeric_limits<uint64_t>::max();
            //std::cout << "Index: " << index << std::endl;
	    //
            profileRequired = false;
        }
        else {
            indices.pop_back();
            if (indices.size() == 0)
                profileRequired = true;
	}
    }

    const KernelConfiguration& getNextConfiguration() const override
    {
        if (indices.size() > 0) {
            std::cout << "getNextConfiguration: index " << indices.back() << " (" << indices.size() << ")\n";
            return getConfigurations().at(indices.back());
        }
        else {
            std::cout << "getNextConfiguration: best index " << bestIdxInBatch << " (" << indices.size() << ")\n";
            return getConfigurations().at(bestIdxInBatch);
        }
    }

    size_t getUnexploredConfigurationCount() const override
    {
        /*if (index >= getConfigurations().size())
        {
            return 0;
        }

        return getConfigurations().size() - index;*/
        //TODO implement it
        return 1;
    }

    bool shouldProfile() {
        return profileRequired;
    }

private:
    std::vector<size_t> indices;
    std::vector<DimensionVector> globalSizes;
    std::vector<DimensionVector> localSizes;
    size_t bestIdxInBatch;
    uint64_t bestBatchDuration;
    double myComputeCapability;
    std::string statPrefix;
    double statComputeCapability;
    std::string scratchPrefix;
    bool profileRequired;
    int pipe_cpp_to_py[2];
    int pipe_py_to_cpp[2];
};

} // namespace ktt
