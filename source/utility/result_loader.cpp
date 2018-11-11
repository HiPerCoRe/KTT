#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <utility/result_loader.h>

namespace ktt
{

bool ResultLoader::loadResults(const std::string& filePath) 
{
    values.clear();

    std::ifstream inputFile(filePath, std::ios::app | std::ios_base::out);

    if (!inputFile.is_open())
    {
        std::cerr << "Unable to open file: " << filePath << std::endl;
        return false;
    }

    std::string line;
    std::vector<std::string> params;
    timeIndex = 0;
    paramsBegin = 0;
    paramsLength = 0;

    /* Read header line */
    std::getline(inputFile, line);
    std::stringstream ss(line);
    while (ss.good())
    {
        std::string substr;
        std::getline(ss, substr, ',');
        params.push_back(substr);
    }

    // backward compatibility, new version produces Computation duration only
    if (params[1] == "Kernel duration (us)")
    {
        timeIndex = 1;
        paramsBegin = 4;
    }
    else if (params[2] == "Kernel duration (us)")
    {
        timeIndex = 2;
        paramsBegin = 5;
    }
    else if (params[1] == "Computation duration (us)")
    {
        timeIndex = 1;
        paramsBegin = 4;
    }
    else
    {
        std::cerr << "Malformed file: " << filePath << std::endl;
        return false;
    }

    // Jump over possible multiple local/global sizes in compositions
    while (params[paramsBegin].compare(0, 10, "Local size") == 0
        || params[paramsBegin].compare(0, 11, "Global size") == 0)
    {
        paramsBegin++;
    }

    paramsLength = params.size() - paramsBegin;
    
    while (std::getline(inputFile, line)) 
    {
        params.clear();
        std::stringstream ss(line);
        while (ss.good())
        {
            std::string substr;
            std::getline(ss, substr, ',');
            params.push_back(substr);
        }
        if (params.size() < paramsBegin+paramsLength)
        {
            std::cerr << "Malformed file: " << filePath << std::endl;
            return false;
        }
        std::vector<int> runValues;
        runValues.push_back(std::stoi(params[timeIndex]));
        for (size_t i = paramsBegin; i < paramsBegin+paramsLength; i++)
            runValues.push_back(std::stoi(params[i]));
        values.push_back(runValues);
    }
    return true;
}

KernelResult ResultLoader::readResult(const KernelConfiguration& configuration) {
    const std::vector<ParameterPair> paramPairs = configuration.getParameterPairs();
    if (paramsLength != paramPairs.size())
    {
        throw std::runtime_error("Number of kernel's tuning parameters mismatch with number of read tuning parameters");
    }

    for (auto val : values)
    {
        bool match = true;
        for (size_t j = 0; j < paramsLength; j++) 
        {
            if (val[j+1] != paramPairs[j].getValue()) 
            {
                match = false;
                break;
            }
        }
        if (match)
        {
//            KernelRunResult krr(val[0], val[0]);
            KernelResult tr("dry run", val[0]);
            return tr;
        }
    }

    /* data not found, suppose kernel failed */
    throw std::runtime_error("Kernel measurement not found");
}

} // namespace ktt
