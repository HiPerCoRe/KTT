#pragma once

#include <algorithm>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <api/searcher/searcher.h>
#include <utility/logger.h>

#define PROFILESEARCHER_TEMPFILE_CONF   "ktt-tempfile-conf.csv"
#define PROFILESEARCHER_TEMPFILE_PC     "ktt-tempfile-pc.csv"
#define PROFILESEARCHER_TEMPFILE_IDX    "ktt-tempfile-idx.dat"
#define NONPROFILE_BATCH 5

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
          Logger::getLogger().log(LoggingLevel::Error, "Profile searcher: Pipes to communicate with Python scripts could not be opened. Quitting.");
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
          std::string command = "python " + scratchPrefix + " ktt-profiling-searcher.py -o " + PROFILESEARCHER_TEMPFILE_CONF + " --oc " + std::to_string(myComputeCapability) + " --mp 46 --co 2944" + " --kb " + statPrefix + "_output_Proposed.sav --ic " + std::to_string(statComputeCapability) + " -i " + std::to_string(bestIdxInBatch) + " -b " + std::to_string(NONPROFILE_BATCH) + " -p " + PROFILESEARCHER_TEMPFILE_PC + " --compute_bound";
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
          Logger::getLogger().log(LoggingLevel::Error, "Profile searcher: Fork of the process to start Python script failed. Quitting.");
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
      Logger::getLogger().log(LoggingLevel::Debug, "Profile searcher writing message to Python script from onReset " + messageToBeSent);
      ::write(pipe_cpp_to_py[1], messageToBeSent.c_str(), messageToBeSent.size());
      //close the pipes
      ::close(pipe_py_to_cpp[0]);
      ::close(pipe_cpp_to_py[1]);
    }

    ~ProfileSearcher()
    {
      //we are done, send message "quit" to python script
      std::string messageToBeSent = "quit";
      Logger::getLogger().log(LoggingLevel::Debug, "Profile searcher writing message to Python script from destructor " + messageToBeSent);
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
            Logger::getLogger().log(LoggingLevel::Debug, "Profile searcher writing message to Python script from calculateNextConfiguration " +  messageToBeSent);
            ::write(pipe_cpp_to_py[1], messageToBeSent.c_str(), messageToBeSent.size());

            // read result of the script
            //create the buffer. the size depends on NONPROFILEBATCH from ktt-profiling-searcher.pyi and some space to breathe is added
            int bufferSize = NONPROFILE_BATCH*sizeof(int)*2;
            std::vector<char> buffer(bufferSize);
            buffer[bufferSize-1] = '\0';
            ::read(pipe_py_to_cpp[0], &(buffer[0]), bufferSize-1);
            std::string bufferString = std::string(&(buffer[0]));

            Logger::getLogger().log(LoggingLevel::Debug, "Profile searcher received message from Python script " + messageToBeSent);
            //parsing the indices from the message
            size_t pos = 0;
            std::string token;
            std::string delimiter = ",";
            std::stringstream ss;
            while ((pos=bufferString.find(delimiter)) != std::string::npos) {
              token = bufferString.substr(0, pos);
              int n;
              try {
                n = std::stoi(token);
                ss << n << " ";
                indices.push_back(n);
              }
              catch (std::invalid_argument) {
                Logger::getLogger().log(LoggingLevel::Error, "Profile searcher is not able to parse message from Python script: " + token);
              }
              bufferString.erase(0, pos+delimiter.length());
            }
            Logger::getLogger().log(LoggingLevel::Debug, "Profile searcher parsed message from Python script " + ss.str());


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
