#include "Api/Output/KernelResult.h"
#include <iomanip>
#include <limits>
#include <string>
#include <iostream>

struct RunStats {
    int totalRuns = 0;
    int successfulRuns = 0;
    double bestDuration = std::numeric_limits<double>::max();
    std::string bestConfig;

    void Update(ktt::KernelResult result) {
        totalRuns++;

        if (result.GetStatus() == ktt::ResultStatus::Ok) {
            successfulRuns++;
            double duration = result.GetTotalDuration();
            if (duration < bestDuration) {
                bestDuration = duration;
                bestConfig = result.GetConfiguration().GetString();
            }
        }
    }

    void Print(const std::string& phaseName, double throughput = -1)
    {
        std::cout << "\n--- " << phaseName << " complete ---" << std::endl;
        std::cout << "Total runs: " << totalRuns << std::endl;
        std::cout << "Successful runs: " << successfulRuns << "/" << totalRuns << std::endl;
        if (!bestConfig.empty()) {
            std::cout << "Best configuration: " << bestConfig << std::endl;
            std::cout << "Best duration: " << bestDuration << " ns" << std::endl;
        }
        if (throughput != -1) 
        {
            std::cout << "Throughput: " << std::fixed << std::setprecision(2) << throughput << " runs/s" << std::endl;
            std::cout.unsetf(std::ios_base::floatfield);
            std::cout.precision(6);
        }
    }
};